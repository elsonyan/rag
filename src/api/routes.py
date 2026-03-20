from fastapi import APIRouter

from src.api.schemas import ChatRequest, ChatResponse, RAGRequest, RAGResponse
from src.services.chat import handle_chat
from src.services.rag import handle_rag

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(**handle_chat(request))


@chat_router.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest) -> RAGResponse:
    return RAGResponse(**handle_rag(request))
