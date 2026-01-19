from elasticsearch import Elasticsearch, BadRequestError

# 1. Connect
print("--- 1. Connecting to ES ---")
client = Elasticsearch("http://localhost:9200")
print(f"Server Info: {client.info()['version']['number']}")

# 2. Test the specific Index Name
index_name = "ec2-index"
print(f"\n--- 2. Checking Index '{index_name}' ---")

try:
    # This is the line failing in your logs (HEAD request)
    exists = client.indices.exists(index=index_name)
    print(f"Index exists? {exists}")
except BadRequestError as e:
    print(f"❌ CRITICAL ERROR on .exists(): {e}")
    print("Possibility: Index name might have hidden characters?")

# 3. Try to Create (with correct V8 syntax)
print(f"\n--- 3. Attempting to Create Index ---")
mapping = {
    "properties": {
        "text": {"type": "text"},
        "metadata": {"type": "object"},
        "vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "cosine"
        }
    }
}

try:
    # Note: Using mappings= arg, NOT body=
    client.indices.create(index=index_name, mappings=mapping)
    print("✅ Index created successfully!")
except BadRequestError as e:
    if "resource_already_exists" in str(e):
        print("ℹ️ Index already exists (This is fine).")
    else:
        print(f"❌ Failed to create index: {e}")
        # Print detailed error reason
        if hasattr(e, 'info'):
            print(f"Detailed Error: {e.info}")

# 4. Clean up (Delete it so you can run the real test)
print(f"\n--- 4. Cleanup ---")
client.indices.delete(index=index_name, ignore_unavailable=True)
print("Index deleted. You are ready to run the worker.")