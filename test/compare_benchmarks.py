import json, time, numpy as np, os, sys, asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.retrievers import BM25Retriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from app.core.embedding import embeddings_base


# --- 1. SETUP & LOADING ---
print("⏳ Loading data...")
with open('data/output/ec2-ug_recursive_medium.json','r') as f:
    raw_data = json.load(f)

docs = [Document(page_content=d['content'], metadata=d['metadata']) for d in raw_data]

# 2. Initialize Pure Vector Retriever (The Baseline)
embedding_model = embeddings_base.sentence_transformer_medium
vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="rag_ec2-ug_recursive_medium", # Ensure this matches your index
    embedding=embedding_model
)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 3. Initialize BM25 (For Hybrid)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 4. Define Helpers
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    fused_score = {}
    for result in results:
        for rank, doc in enumerate(result):
            doc_str = doc.page_content
            if doc_str not in fused_score:
                fused_score[doc_str] = {"doc": doc, "score": 0.0}
            fused_score[doc_str]["score"] += 1.0 / (rank + k)
    reranked = sorted(fused_score.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked]

def calculate_token_recall(target, retrieved_text):
    target_tokens = set(target.lower().split())
    retrieved_tokens = set(retrieved_text.lower().split())
    if not target_tokens: return 0
    return len(target_tokens.intersection(retrieved_tokens)) / len(target_tokens)

def evaluate_single_pass(docs, target):
    for rank, doc in enumerate(docs):
        if calculate_token_recall(target, doc.page_content) > 0.7:
            return 1, 1 / (rank + 1) # Hit, MRR
    return 0, 0


async def get_hybrid_results(query):
    # Fire both requests at the same time
    # 'await asyncio.gather' runs them in parallel
    vec_results, kw_results = await asyncio.gather(
        vector_retriever.ainvoke(query),  # Note: ainvoke (async invoke)
        bm25_retriever.ainvoke(query)
    )
    
    # Fuse them once both arrive
    return reciprocal_rank_fusion([vec_results, kw_results])

# --- 5. MAIN BENCHMARK LOOP ---
async def main():
    with open('benchmark/gold/ec2-ug_gold_responses.json','r') as f:
        queries = json.load(f)

    stats = {
        "Vector": {"hits": [], "mrr": [], "latencies": []},
        "Hybrid": {"hits": [], "mrr": [], "latencies": []}
    }

    print(f"🚀 Starting Head-to-Head Benchmark on {len(queries)} queries...\n")

    for i, item in enumerate(queries):
        query = item['question']
        target = item.get('ground_truth') or item.get('context')

        # --- A. Test Pure Vector ---
        t0 = time.time()
        vec_docs = vector_retriever.invoke(query)
        stats["Vector"]["latencies"].append(time.time() - t0)
        
        hit, mrr = evaluate_single_pass(vec_docs, target)
        stats["Vector"]["hits"].append(hit)
        stats["Vector"]["mrr"].append(mrr)

        # --- B. Test Hybrid ---
        t0 = time.time()
        # Note: We reuse vec_docs to save time, mimicking a real app where you'd run parallel
        hybrid_docs = await get_hybrid_results(query)
        stats["Hybrid"]["latencies"].append(time.time() - t0)

        hit, mrr = evaluate_single_pass(hybrid_docs, target)
        stats["Hybrid"]["hits"].append(hit)
        stats["Hybrid"]["mrr"].append(mrr)

        if (i+1) % 10 == 0: print(f"Processed {i+1}/{len(queries)}...")

    # --- 6. PRINT FINAL REPORT ---
    print("\n" + "="*45)
    print(f"{'METRIC':<15} | {'VECTOR (Base)':<12} | {'HYBRID (New)':<12}")
    print("="*45)
    print(f"{'Hit Rate':<15} | {np.mean(stats['Vector']['hits']):.2%}       | {np.mean(stats['Hybrid']['hits']):.2%}")
    print(f"{'MRR':<15} | {np.mean(stats['Vector']['mrr']):.3f}        | {np.mean(stats['Hybrid']['mrr']):.3f}")
    print(f"{'Avg Latency':<15} | {np.mean(stats['Vector']['latencies']):.4f}s      | {np.mean(stats['Hybrid']['latencies']):.4f}s")
    print("="*45)

if __name__ == "__main__":
    asyncio.run(main())