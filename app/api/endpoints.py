import os
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.ingest import process_document_task, generate_vectors,celery_app

class EmbeddingRequest(BaseModel):
    filename: str
    embedding_size: str 
    index_name: str

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

@router.post("/embed/{index_name}")
async def embed_existing_document(request: EmbeddingRequest):

    file_path = OUTPUT_DIR / request.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {request.filename} not found.")
    result = generate_vectors.delay(request.filename, request.embedding_size, request.index_name)

    return {
        "message": "Embedding process started.",
        "details": result.id,
        "target_index": request.index_name
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