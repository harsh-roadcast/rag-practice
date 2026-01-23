import argparse
import os
import sys
import json
import time
import pandas as pd
import warnings
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

# Filter warnings
warnings.filterwarnings("ignore")

# Import App Services
from app.services.ingest import process_document_task, generate_vectors, celery_app
from app.api.endpoints import parse_chunk_metadata

# Import LangChain / Ragas
from langchain_openai import ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate
from datasets import Dataset

# Force Celery to run synchronously (no worker needed)
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

load_dotenv()

# --- CONSTANTS ---
PDF_SOURCE = "data/manuals/aws-overview.pdf"
GOLD_DATA_FILE = "benchmark/gold/aws-overview_gold_responses.json"
RESULTS_FILE = "benchmark/results/final_benchmark_results.csv"

EMBEDDING_MODELS = {
    "small": "sentence-transformers/all-MiniLM-L6-v2",
    "medium": "BAAI/bge-base-en-v1.5",
    "large": "BAAI/bge-large-en-v1.5"
}

metrics_list = [
    Faithfulness(),
    AnswerRelevancy()
]

def ensure_data_exists(strategy, size):
    """
    Ensures the chunk JSON and Elasticsearch Index exist.
    """
    # 1. Check for JSON file
    # Construction logic matches ingest.py: f"{base_name}_{strategy}_{embedding_size}.json"
    base_name = "aws-overview" # Derived from filename
    expected_filename = f"{base_name}_{strategy}_{size}.json"
    expected_path = os.path.join("data/output", expected_filename)
    
    if not os.path.exists(expected_path):
        print(f"⚙️ [Setup] Chunk file {expected_filename} missing. Generating...")
        # Call ingestion synchronously
        result = process_document_task.apply(args=[PDF_SOURCE, strategy, size]).result
        if result.get("status") != "success":
            raise Exception(f"Ingestion failed: {result}")
        print("   ✅ Chunking complete.")
    else:
        print(f"   ✅ Found chunk file: {expected_filename}")

    # 2. Re-Index to Elasticsearch (Always rename/reindex to ensure chunk_id presence)
    print(f"⚙️ [Setup] Updating Index for {strategy}-{size}...")
    try:
        # We invoke generate_vectors.apply() to run it locally
        res = generate_vectors.apply(args=[expected_filename], kwargs={"embedding_size": size, "strategy": strategy})
        # Note: generate_vectors now checks/creates index and uploads
        print("   ✅ Indexing complete.")
    except Exception as e:
        print(f"   ❌ Indexing failed: {e}")
        raise e
        
    return f"rag_{base_name}_{strategy}_{size}"

def run_benchmark_for_config(strategy, size, index_name):
    print(f"\n🚀 [Benchmarking] Strategy: {strategy} | Size: {size}")
    
    # Load Gold Data
    with open(GOLD_DATA_FILE, "r") as f:
        gold_data = json.load(f)
        
    # Setup Retriever
    model_name = EMBEDDING_MODELS[size]
    print(f"   Loading Embedding Model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vectorstore = ElasticsearchStore(
        es_url=os.getenv("ES_URL", "http://localhost:9200"),
        index_name=index_name,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Setup Generator
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on context:\n{context}\n\nQ: {question}""")
    
    # Storage
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": [], "retrieval_latency": [], "generation_latency": []}
    hit_count = 0
    
    total_q = len(gold_data)
    print(f"   Running {total_q} queries...")
    
    for i, item in enumerate(gold_data):
        q = item["question"]
        target_id = str(item.get("source_chunk_id"))
        
        # 1. Retrieval
        t0 = time.perf_counter()
        try:
            docs = retriever.invoke(q)
        except Exception as e:
            print(f"   ❌ Retrieval failed for Q{i}: {e}")
            continue
        t1 = time.perf_counter()
        
        # 2. Hit Rate Check
        # Note: Hit Rate is only strictly valid if the strategy matches the gold data's strategy (recursive)
        # For 'length' strategy, IDs won't match, so Hit Rate will naturally be 0.
        retrieved_ids = [str(d.metadata.get("chunk_id")) for d in docs]
        if target_id in retrieved_ids:
            hit_count += 1
            
        # 3. Generation
        context_text = "\n".join([d.page_content for d in docs])
        t2 = time.perf_counter()
        ans = llm.invoke(prompt.invoke({"context": context_text, "question": q})).content
        t3 = time.perf_counter()
        
        ragas_data["question"].append(q)
        ragas_data["answer"].append(ans)
        ragas_data["contexts"].append([d.page_content for d in docs])
        ragas_data["ground_truth"].append(item["ground_truth"])
        ragas_data["retrieval_latency"].append(t1 - t0)
        ragas_data["generation_latency"].append(t3 - t2)
        
        if (i+1) % 10 == 0:
            print(f"     Processed {i+1}/{total_q}...")

    # Ragas Eval
    print("   📊 Calculating Ragas Metrics...")
    dataset = Dataset.from_dict({
        k: v for k, v in ragas_data.items() 
        if k in ["question", "answer", "contexts", "ground_truth"]
    })
    
    scores = evaluate(
        dataset,
        metrics=metrics_list,
        llm=ChatOpenAI(model="gpt-4.1-mini"),
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS["small"])
    )
    
    # Aggregate
    result_row = {
        "strategy": strategy,
        "embedding_size": size,
        "hit_rate": hit_count / total_q,
        "avg_retrieval_sec": sum(ragas_data["retrieval_latency"]) / total_q,
        "avg_gen_sec": sum(ragas_data["generation_latency"]) / total_q,
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"]
    }
    
    # Save to CSV (Appended)
    df = pd.DataFrame([result_row])
    header = not os.path.exists(RESULTS_FILE)
    df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
    print(f"✅ Results saved for {strategy}-{size}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--size", required=True)
    args = parser.parse_args()
    
    try:
        idx_name = ensure_data_exists(args.strategy, args.size)
        run_benchmark_for_config(args.strategy, args.size, idx_name)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
