from elasticsearch import Elasticsearch, helpers

class VectorDB:
    def __init__(self):
        # Initialize the Elasticsearch client
        self.es_client: Elasticsearch = Elasticsearch(hosts=["http://localhost:9200"])

    def create_rag_index(self, index_name: str, vector_dims: int = 768):
        """
        Creates an Elasticsearch index for RAG if it doesn't already exist.
        """
        if self.es_client.indices.exists(index=index_name):
            print(f"vector_db: Index '{index_name}' already exists.")
            return

        # FIXED: Removed the outer "mappings": { ... } wrapper
        # The client now expects just the properties dictionary directly.
        index_mapping = {
            "properties": {
                # For keyword search
                "text": {"type": "text"},
                # For metadata filtering
                "metadata": {"type": "object"},
                # For vector search
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dims,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }

        # FIXED: Use 'mappings=' instead of 'body='
        try:
            self.es_client.indices.create(index=index_name, mappings=index_mapping)
            print(f"vector_db: ✅ Created index '{index_name}' successfully.")
        except Exception as e:
            print(f"vector_db: ❌ Error creating index: {e}")
            raise e

    def delete_rag_index(self, index_name: str):
        """
        Deletes an Elasticsearch index for RAG if it exists.
        """
        if self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)
            print(f"Deleted index '{index_name}' successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")

    def upload_chunks(self, index_name: str, chunks: list, vectors: list):
        actions = []
        for i, chunk in enumerate(chunks):
            doc = {
                "_index": index_name,
                "_source": {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                    "vector": vectors[i]
                }
            }
            actions.append(doc)

        # Bulk upload
        success, errors = helpers.bulk(self.es_client, actions)
        print(f"vector_db: 🚀 Inserted {success} documents into index '{index_name}'.")
        return success

# Initialize global instance
vector_db = VectorDB()