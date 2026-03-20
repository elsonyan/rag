from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.knowledge_routes import router as knowledge_router
from src.api.routes import chat_router
from src.api.schemas import HealthResponse
from src.core import config as cfg


def create_app() -> FastAPI:
    cfg.ensure_directories()
    app = FastAPI(title="LLM & RAG API")
    allow_origins = cfg.allowed_origins or ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials="*" not in allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(knowledge_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            knowledge_file_folder=cfg.knowledge_file_folder,
            chroma_folder=cfg.chroma_folder,
        )

    return app


app = create_app()
