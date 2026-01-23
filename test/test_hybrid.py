import json, time, numpy as np, os, sys
from difflib import SequenceMatcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.retrievers import BM25Retriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from app.core.embedding import embeddings_base


with open('data/output/ec2-ug_recursive_medium.json','r') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} documents for indexing.")

docs = [Document(page_content=data['content'],metadata=data['metadata']) for data in raw_data]

embedding_model = embeddings_base.sentence_transformer_medium

vector_retriever = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="rag_ec2-ug_recursive_medium",
    embedding=embedding_model
).as_retriever(search_kwargs={"k": 5})

print("Vector retriever initialized.")


bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

print("BM25 retriever initialized.")

def reciprocal_rank_fusion(results: list[list[Document]],k=60):
    """  Perform Reciprocal Rank Fusion on multiple retrieval results. """
    fused_score = {}

    for result in results:
        for rank, doc in enumerate(result):
            doc_str = str(doc.page_content)
            if doc_str not in fused_score:
                fused_score[doc_str] = {"doc":doc,"score":0.0}
            
            fused_score[doc_str]["score"] += 1 / (rank + k)

    reranked = sorted(fused_score.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked]

# Define a Token Recall helper
def calculate_token_recall(target_text, retrieved_text):
    # simple splitting by whitespace and lowercase
    target_tokens = set(target_text.lower().split())
    retrieved_tokens = set(retrieved_text.lower().split())
    
    # How many target words are in the retrieved text?
    intersection = target_tokens.intersection(retrieved_tokens)
    
    if len(target_tokens) == 0: return 0
    return len(intersection) / len(target_tokens)

with open('benchmark/gold/ec2-ug_gold_responses.json','r') as f:
    gold_data = json.load(f)

print(f"Loaded {len(gold_data)} gold standard queries.")

queries = [item for item in gold_data]

hits = []
reciprocal_ranks = []
latencies = []

for i,item in enumerate(queries):
    query = item['question']
    target_answer = item.get('ground_truth')

    start_time = time.time()
    vec_results = vector_retriever.invoke(query)
    kw_results = bm25_retriever.invoke(query)
    hybrid_results = reciprocal_rank_fusion([vec_results, kw_results])
    end_time = time.time()
    latencies.append(end_time - start_time)


    found = False
    for rank, doc in enumerate(hybrid_results):
        similarity = calculate_token_recall(target_answer, doc.page_content)
        if similarity > 0.7 :
            hits.append(1)
            reciprocal_ranks.append(1 / (rank + 1))
            found = True
            break

    if not found:
        hits.append(0)
        reciprocal_ranks.append(0)

        if len(hits) < 4:
            print(f"\n❌ MISS on: '{query}'")
            print(f"   EXPECTED: '{target_answer[:60]}...'")
            print(f"   GOT TOP 1: '{hybrid_results[0].page_content[:60]}...'")

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(queries)} queries")

print("\n--- 📊 Hybrid Search Results ---")
print(f"✅ Hit Rate: {np.mean(hits):.2%} (Target: >85%)")
print(f"🏅 MRR Score: {np.mean(reciprocal_ranks):.3f} (Target: >0.70)")
print(f"⏱️ Avg Latency: {np.mean(latencies):.4f}s (Target: <0.3s)")