from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    strategy: str = "recursive"   # recursive, semantic, length
    embedding_size: str = "medium" # small, medium, large

class QueryResponse(BaseModel):
    answer: str
    sources: list