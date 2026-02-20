from typing import List, Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
