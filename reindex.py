import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.services.ingest import generate_vectors, celery_app

# Force synchronous execution so we don't need a worker running
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

if __name__ == "__main__":
    target_file = "rag_ec2-ug_recursive_small.json"
    print(f"🔄 Re-indexing {target_file} to Elasticsearch...")
    print("   (This helps fix the missing chunk_id issue for benchmarking)")
    
    try:
        # Call the task synchronously
        # Using .apply() is the standard way to execute a task immediately in the current process
        task_result = generate_vectors.apply(args=[target_file], kwargs={"embedding_size": "small", "strategy": "recursive"})
        
        # Result content depends on what generate_vectors returns
        print("\n✅ Indexing Complete!")
        print(f"Result: {task_result.result}")
        
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
