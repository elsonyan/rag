from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    messages: list[str] = Field(..., min_length=1)
    session_id: str | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[str]) -> list[str]:
        cleaned_messages = [message.strip() for message in messages if message and message.strip()]
        if not cleaned_messages:
            raise ValueError("messages must contain at least one non-empty message")
        return cleaned_messages


class RAGRequest(BaseModel):
    query: str
    session_id: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        return cleaned_query


class ChatResponse(BaseModel):
    response: str


class RAGResponse(BaseModel):
    response: str
    sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    knowledge_file_folder: str
    chroma_folder: str


class KnowledgeOverviewResponse(BaseModel):
    knowledge_dir: str
    file_count: int
    supported_file_count: int
    total_size_bytes: int
    files: list[str] = Field(default_factory=list)
