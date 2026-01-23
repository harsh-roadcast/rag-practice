from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

class EmbeddingRequest(BaseModel):
    filename: str