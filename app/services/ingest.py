import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from celery import Celery
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from app.core.chunking import chunker_service
from app.core.embedding import embeddings_base
from app.services.vector_db import vector_db  # Import the vector_db module

# Initialize Celery
celery_app = Celery(
    "ingest",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Configure Celery to use spawn instead of fork for macOS compatibility
celery_app.conf.update(
    worker_pool='solo',  # Use solo pool for macOS compatibility
)

# Define where to save the JSON files
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INDEX_REGISTRY_PATH = os.path.join(OUTPUT_DIR, "index_registry.json")


def _load_index_registry() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(INDEX_REGISTRY_PATH):
        return {}
    with open(INDEX_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index_registry(registry: Dict[str, Dict[str, str]]) -> None:
    with open(INDEX_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)


def record_index_mapping(strategy: str, embedding_size: str, index_name: str, source_file: str) -> None:
    registry = _load_index_registry()
    key = f"{strategy}:{embedding_size}"
    registry[key] = {
        "index_name": index_name,
        "source_file": source_file,
        "strategy": strategy,
        "embedding_size": embedding_size,
        "updated_at": datetime.utcnow().isoformat()
    }
    _save_index_registry(registry)



def _infer_strategy_from_filename(json_filename: str) -> str:
    stem = Path(json_filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[-2]
    return "recursive"

def save_chunks_to_json(chunks, filename):
    """
    Helper to save a list of Document objects to a JSON file.
    """
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return filepath

def parse_chunk_metadata(json_filename:str)-> tuple[str,str]:
    """ Extract strategy and embedding size from filename """
    stem = Path(json_filename).stem
    parts = stem.split("_")
    if len(parts)>=2:
        return parts[-2], parts[-1]
    raise ValueError(f"Filename {json_filename} doesn't match pattern: name_strategy_size.json")

@celery_app.task
def generate_vectors(json_filename: str, embedding_size: str = "medium", strategy: Optional[str] = None):
    """
    Docstring for generate_vectors
    
    :param json_filename: 
    :type json_filename: str
    :param embedding_size: Description
    :type embedding_size: str
    """
    # Generate file path
    file_path = os.path.join(OUTPUT_DIR, json_filename)
    # Check if file exists
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File {json_filename} does not exist."}

    print(f"worker: Generating vectors for {json_filename} using {embedding_size} embeddings...")

    with open(file_path,"r", encoding="utf-8") as f:
        data = json.load(f)

    if embedding_size == "small":
        embeddings = embeddings_base.sentence_transformer_small
        dims = 384
    elif embedding_size == "medium":
        embeddings = embeddings_base.sentence_transformer_medium
        dims = 768
    else:
        embeddings = embeddings_base.sentence_transformer_large
        dims = 1024

    strategy = strategy or _infer_strategy_from_filename(json_filename)
    index_name = f"rag_{Path(json_filename).stem}"

    print(f"worker: checked/created index {index_name} with dims {dims}")
    
    try:
        vector_db.create_rag_index(index_name=index_name, vector_dims=dims)
    except Exception as e:
        return {"status": "error", "message": f"Could not connect to Elasticsearch: {str(e)}"}

    
    vectors = []
    batch_size = 1  # Process 1 chunk at a time (Safe for CPU)
    total_chunks = len(data)
    print(f"worker: Processing {total_chunks} chunks in batches of {batch_size}...")

    total_uploaded = 0

    for i in range(0, total_chunks, batch_size):
        batch_items = data[i : i + batch_size]
        
        # Embed just this small batch
        batch_texts = [item["content"] for item in batch_items]
        # Fixed: Inject chunk_id into metadata so it is indexed in Elasticsearch
        batch_docs = []
        for item in batch_items:
            meta = item["metadata"].copy()
            meta["chunk_id"] = item["chunk_id"]
            batch_docs.append(Document(page_content=item["content"], metadata=meta))

        try:
            vectors = embeddings.embed_documents(batch_texts)
        except Exception as e:
            return {"status": "error", "message": f"Embedding error: {str(e)}"}
            

    
        vector_db.upload_chunks(index_name, batch_docs, vectors)

        total_uploaded += len(batch_docs)
        print(f"worker: Uploaded {total_uploaded} / {total_chunks} chunks so far.")
   

    record_index_mapping(strategy=strategy, embedding_size=embedding_size, index_name=index_name, source_file=json_filename)

    return {
        "status": "success",
        "index_name": index_name,
        "counts_uploaded": total_uploaded
    }




@celery_app.task
def process_document_task(file_path: str, strategy: str = "recursive", embedding_size: str = "medium"):
    """
    Background task to load PDF, chunk it, and save to JSON.
    """
    print(f"worker: Processing {file_path} with strategy={strategy}...")

    # 1. Load the PDF
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.lazy_load()
        print(f"worker: PDFLoader Initialized (Lazy mode)")
    except Exception as e:
        return f"Error loading PDF: {str(e)}"

    # 2. Chunk the text
    if strategy == "recursive":
        chunks = chunker_service.chunk_text_recursive(pages)
    elif strategy == "length":
        chunks = chunker_service.chunk_text_length_based(pages)
    elif strategy == "semantic":
        chunks = chunker_service.chunk_text_semantic(pages, embedding_type=embedding_size)
    else:
        return "Invalid strategy selected."

    print(f"worker: Created {len(chunks)} chunks.")

    # 3. SAVE TO JSON (New Step)
    # Create a unique filename based on the input file and strategy
    base_name = os.path.basename(file_path).replace(".pdf", "")
    json_filename = f"{base_name}_{strategy}_{embedding_size}.json"
    
    saved_path = save_chunks_to_json(chunks, json_filename)
    print(f"worker: Saved chunks to {saved_path}")

    return {
        "status": "success", 
        "file": file_path, 
        "chunks_created": len(chunks),
        "strategy": strategy,
        "json_output": saved_path
    }

