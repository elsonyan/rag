from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(env_name: str, default: Path) -> str:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return str(default)

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
llm_base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
llm_model = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
silicon_embedding_base_url = os.getenv(
    "SILICON_EMBEDDING_BASE_URL",
    "https://api.siliconflow.cn/v1/embeddings",
).rstrip("/")
silicon_embedding_model = os.getenv(
    "SILICON_EMBEDDING_MODEL",
    "netease-youdao/bce-embedding-base_v1",
)
chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "100"))
chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "10"))
similarity_search_k = int(os.getenv("RAG_SIMILARITY_SEARCH_K", "30"))
re_ranker_model = os.getenv("RERANKER_MODEL", "netease-youdao/bce-reranker-base_v1")
re_ranker_url = os.getenv("RERANKER_URL", "https://api.siliconflow.cn/v1/rerank").rstrip("/")
re_ranker_k = int(os.getenv("RAG_RERANK_TOP_K", "3"))

src_dir = _resolve_path("RAG_SRC_DIR", PROJECT_ROOT / "src")
data_dir = _resolve_path("RAG_DATA_DIR", PROJECT_ROOT / "data")
storage_dir = _resolve_path("RAG_STORAGE_DIR", Path(data_dir) / "storage")
bloom_folder = _resolve_path("RAG_BLOOM_DIR", Path(storage_dir) / "bloom")
chroma_folder = _resolve_path("RAG_CHROMA_DIR", Path(storage_dir) / "chroma")
knowledge_file_folder = _resolve_path("RAG_KNOWLEDGE_DIR", Path(data_dir) / "knowledge_files")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]


def ensure_directories() -> None:
    for folder in (Path(bloom_folder), Path(chroma_folder), Path(knowledge_file_folder)):
        folder.mkdir(parents=True, exist_ok=True)
