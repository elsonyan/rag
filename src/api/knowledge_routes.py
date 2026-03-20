from fastapi import APIRouter

from src.api.schemas import KnowledgeOverviewResponse
from src.services.knowledge import get_knowledge_overview

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/overview", response_model=KnowledgeOverviewResponse)
def knowledge_overview() -> KnowledgeOverviewResponse:
    return KnowledgeOverviewResponse(**get_knowledge_overview())
