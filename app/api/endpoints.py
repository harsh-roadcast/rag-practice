import os
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel


from app.services.ingest import (
    parse_chunk_metadata,
    process_document_task,
    generate_vectors,
    celery_app,
    _load_index_registry,

)
from app.core.config import QueryRequest, QueryResponse, EmbeddingRequest
from app.core.embedding import embeddings_base
from app.services.vector_db import vector_db
from app.services.llm_service import get_llm_service
    

router = APIRouter()

# Ensure we have a place to store uploaded files
UPLOAD_DIR = Path("data/uploaded_manuals")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),   # Default to recursive
    vector_size: str = Form("medium")    # Default to medium
):
    """
    Uploads a PDF and triggers the background ingestion task.
    """
    # 1. Save the file locally first
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    safe_filename = os.path.basename(file.filename)
    file_path = UPLOAD_DIR / safe_filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Trigger the Celery task (The .delay() method sends it to Redis)
    task = process_document_task.delay(str(file_path), strategy, vector_size)

    return {
        "message": "Ingestion started successfully.",
        "task_id": task.id,
        "filename": safe_filename,
        "strategy_used": strategy
    }

@router.get("/status/{task_id}")
def get_status(task_id: str):
    """
    Check if the ingestion is finished.
    """
    result = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result
    }

@router.post("/embed")
async def embed_existing_document(request: EmbeddingRequest):

    file_path = OUTPUT_DIR / request.filename

    strategy, embedding_size = parse_chunk_metadata(request.filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {request.filename} not found.")
    if embedding_size not in {"small", "medium", "large"}:
        raise HTTPException(status_code=400, detail="Invalid embedding size. Choose from 'small', 'medium', 'large'.")
    if strategy not in {"recursive", "semantic", "length"}:
        raise HTTPException(status_code=400, detail="Invalid strategy. Choose from 'recursive', 'semantic', 'length'.")

    target_index = f"rag_{Path(request.filename).stem}"
    result = generate_vectors.delay(request.filename, embedding_size, strategy)

    return {
        "message": "Embedding process started.",
        "details": result.id,
        "target_index": target_index
    }

@router.get("/list-chunks")
def list_uploaded_chunks():
    """
    Lists all uploaded documents available for embedding.
    """
    files = os.listdir(UPLOAD_DIR)
    return {
        "uploaded_files": files
    }

@router.post("/query", response_model=QueryResponse)
async def process_query_rag(request: QueryRequest):
    """
    Process a user query using RAG with the specified chunking strategy.
    """

    print(f"API Receieved query: {request.question} ") 

    registry = _load_index_registry()
    if not registry:
        raise HTTPException(status_code=404,detail="No indices found in registry. Please embed documents first.")
    


    latest_mapping = list(registry.values())[-1]
    stratergy,embedding_size = latest_mapping["strategy"],latest_mapping["embedding_size"]
    if not latest_mapping:
        raise HTTPException(
            status_code=404,
            detail="No embedded index found for the selected strategy and embedding size. Run /embed first."
        )


    if embedding_size == "small":
        query_vector = embeddings_base.sentence_transformer_small.embed_query(request.question)
    elif embedding_size == "medium":
        query_vector = embeddings_base.sentence_transformer_medium.embed_query(request.question)
    else:
        query_vector = embeddings_base.sentence_transformer_large.embed_query(request.question)

    all_results = []
    for key,mapping in registry.items():
        index_name = mapping["index_name"]
        try:
            results = vector_db.search(index_name=index_name, query_vector=query_vector, top_k=5)
            all_results.extend(results)
        except Exception as e:
            print(f"API: ❌ Error searching index '{index_name}': {e}")
    
    if not all_results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found in the vector database."
        )
    
    all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)


    print(f"API: Retrieved {len(all_results)} chunks from all indices.")

    context_chunks = []
    for chunk in all_results:
        text = chunk.get("text") or chunk.get("content") or chunk.get("page_content")
        if not text:
            continue
        context_chunks.append({
            "text": text,
            "metadata": chunk.get("metadata", {})
        })

    if not context_chunks:
        raise HTTPException(
            status_code=404,
            detail="Retrieved chunks did not contain any text content to feed the LLM."
        )

    answer = get_llm_service().get_answer(request.question, context_chunks)
    sources = []
    for chunk in context_chunks:
        metadata = dict(chunk["metadata"])
        metadata.pop("keywords", None)
        sources.append(metadata)
    return {
        "answer": answer,
        "sources": sources
    }
    