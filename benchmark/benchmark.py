import time
import json
import os
import pandas as pd
import warnings
from dotenv import load_dotenv

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

# Updated imports for compatibility
from langchain_openai import ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Ragas Imports
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate
from datasets import Dataset

load_dotenv()

# --- CONFIGURATION ---
GOLD_DATA_FILE = "benchmark/gold/aws-overview_gold_responses.json"
RESULTS_FILE = "benchmark/results/elastic_benchmark_results.csv"
INDEX_REGISTRY_PATH = "data/output/index_registry.json"

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "changeme")

EMBEDDING_MODELS = {
    "small": "sentence-transformers/all-MiniLM-L6-v2",
    "medium": "BAAI/bge-base-en-v1.5",
    "large": "BAAI/bge-large-en-v1.5"
}

metrics_list = [
    Faithfulness(),
    AnswerRelevancy()
]

def load_configurations_from_registry():
    """
    Dynamically loads available indices from the registry file.
    """
    configs = []
    if not os.path.exists(INDEX_REGISTRY_PATH):
        print(f"⚠️ Warning: Registry file {INDEX_REGISTRY_PATH} not found.")
        return configs
        
    with open(INDEX_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    for key, data in registry.items():
        # key format is typically "strategy:size"
        strat = data.get("strategy", "unknown")
        size = data.get("embedding_size", "medium")
        index = data.get("index_name")
        
        # Create a config entry compatible with our benchmark loop
        configs.append({
            "id": f"{strat}_{size}",
            "strategy": strat,
            "model": size,
            "index_name": index
        })
    return configs

# --- HELPER FUNCTIONS ---

def get_retriever(config):
    """
    Connects to the specific Elasticsearch Index for the configuration.
    """
    model_name = EMBEDDING_MODELS.get(config["model"], EMBEDDING_MODELS["medium"])
    
    # Use CPU for benchmark stability/compatibility if CUDA not forced
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    try:
        # Connect to existing index
        vectorstore = ElasticsearchStore(
            es_url=ES_URL,
            index_name=config["index_name"],
            embedding=embeddings,
            # es_user=ES_USERNAME, # Uncomment if security enabled
            # es_password=ES_PASSWORD,
        )
        
        # Return retriever (k=5 matched generate_gold)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
        
    except Exception as e:
        print(f"⚠️ ERROR connecting to index {config['index_name']}: {e}")
        return None

def load_gold_dataset():
    if not os.path.exists(GOLD_DATA_FILE):
        print(f"❌ Gold dataset not found at {GOLD_DATA_FILE}. Please run generate_gold.py first.")
        return []
    with open(GOLD_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- MAIN BENCHMARK LOOP ---

def run_benchmark():
    gold_data = load_gold_dataset()
    if not gold_data:
        return

    # Load Configs Dynamically
    configurations = load_configurations_from_registry()
    if not configurations:
        print("❌ No configurations found in registry. Have you run ingestion?")
        return
        
    all_results = []
    
    # Initialize LLM (GPT-4.1-mini for cost savings) for answer generation
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    print(f"\n🚀 Found {len(configurations)} indices to benchmark.")

    for config in configurations:
        print(f"\n---------------------------------------------------------")
        print(f"🚀 Benchmarking Config: {config['id']}")
        print(f"   Index: {config['index_name']}")
        print(f"---------------------------------------------------------")
        
        retriever = get_retriever(config)
        if not retriever:
            print("   Skipping (Retriever init failed)...")
            continue
            
        # Ragas Storage
        ragas_data = {
            "question": [], "answer": [], "contexts": [], 
            "ground_truth": [], "retrieval_latency": [], "generation_latency": []
        }
        
        hit_count = 0
        
        # Loop through questions
        for idx, item in enumerate(gold_data):
            question = item["question"]
            ground_truth = item["ground_truth"]
            # Gold data stores 'source_chunk_id' as int or str, normalize to compare
            target_chunk_id = str(item.get("source_chunk_id"))
            
            # 1. Measure Retrieval Latency
            t0 = time.perf_counter()
            try:
                retrieved_docs = retriever.invoke(question)
            except Exception as e:
                print(f"   ❌ Retrieval failed for Q{idx}: {e}")
                continue
            t1 = time.perf_counter()
            retrieval_time = t1 - t0
            
            # 2. Check Hit Rate
            # We compare stringified IDs to be safe
            retrieved_ids = [str(doc.metadata.get("chunk_id", "None")) for doc in retrieved_docs]
            
            # Debug log for first item if missing
            if idx == 0 and "None" in retrieved_ids and config["id"] == configurations[0]["id"]:
                print("   ⚠️ WARNING: chunk_id missing in retrieved docs. Hit Rate will be 0.")
                print("   (Did you re-run ingestion after updating ingest.py?)")

            is_hit = target_chunk_id in retrieved_ids
            if is_hit:
                hit_count += 1
                
            # 3. Measure Generation Latency
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            chain_input = {"context": context_text, "question": question}
            
            t2 = time.perf_counter()
            try:
                formatted_prompt = prompt.invoke(chain_input)
                answer = llm.invoke(formatted_prompt).content
            except Exception as e:
                print(f"   ❌ Generation failed for Q{idx}: {e}")
                answer = "Error generating answer"
            t3 = time.perf_counter()
            gen_time = t3 - t2
            
            # Store Data
            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append([doc.page_content for doc in retrieved_docs])
            ragas_data["ground_truth"].append(ground_truth)
            ragas_data["retrieval_latency"].append(retrieval_time)
            ragas_data["generation_latency"].append(gen_time)
            
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx+1}/{len(gold_data)} queries...")

        # 4. Ragas Evaluation
        print(f"📊 Running Ragas Eval for {config['id']}...")
        
        if len(ragas_data["question"]) == 0:
            print("   No data collected, skipping Ragas.")
            continue

        dataset = Dataset.from_dict({
            "question": ragas_data["question"],
            "answer": ragas_data["answer"],
            "contexts": ragas_data["contexts"],
            "ground_truth": ragas_data["ground_truth"]
        })
        
        eval_llm = ChatOpenAI(model="gpt-4.1-mini")
        
        try:
            results = evaluate(
                dataset,
                metrics=metrics_list,
                llm=eval_llm,
                embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS["small"]) 
            )
            # Ragas 0.4.x Result access
            faithfulness_score = results["faithfulness"]
            relevancy_score = results["answer_relevancy"]
        except Exception as e:
            print(f"   ❌ Ragas evaluation failed: {e}")
            faithfulness_score = 0.0
            relevancy_score = 0.0
        
        # 5. Aggregate & Save
        avg_ret_latency = sum(ragas_data["retrieval_latency"]) / max(len(ragas_data["retrieval_latency"]), 1)
        avg_gen_latency = sum(ragas_data["generation_latency"]) / max(len(ragas_data["generation_latency"]), 1)
        hit_rate = hit_count / max(len(gold_data), 1)
        
        final_score = {
            "config_id": config["id"],
            "index": config["index_name"],
            "hit_rate": hit_rate,
            "avg_retrieval_latency": avg_ret_latency,
            "avg_generation_latency": avg_gen_latency,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score
        }
        
        all_results.append(final_score)
        
        df = pd.DataFrame(all_results)
        df.header = True
        df.to_csv(RESULTS_FILE, index=False, mode='a', header=not os.path.exists(RESULTS_FILE))
        print(f"✅ Saved results for {config['id']}")

    print("\n🏆 Benchmark Complete!")
    if all_results:
        print(pd.DataFrame(all_results))

if __name__ == "__main__":
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    # Clear previous results if starting fresh
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    run_benchmark()
