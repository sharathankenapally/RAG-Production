from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    session_id: Optional[str] = None
    filters: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    cached: bool
    evaluation: Optional[dict] = None
