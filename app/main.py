from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="RAG Benchmark API")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "running"}