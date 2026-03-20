from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from src.clients.siliconflow_embeddings import SiliconFlowEmbeddings
from src.core import config as cfg


class LazyEmbeddingProxy(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_embedding_model().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return get_embedding_model().embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> SiliconFlowEmbeddings:
    if not cfg.api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not configured.")

    return SiliconFlowEmbeddings(
        model=cfg.silicon_embedding_model,
        api_key=cfg.api_key,
        base_url=cfg.silicon_embedding_base_url,
    )


_embedding_proxy = LazyEmbeddingProxy()


def get_embedding_proxy() -> Embeddings:
    return _embedding_proxy


def embedding_text(text: Optional[str | List[str]]) -> List[List[float]]:
    embedding_model = get_embedding_model()
    if isinstance(text, str):
        return [embedding_model.embed_query(text)]
    if isinstance(text, list):
        return embedding_model.embed_documents(text)
    raise TypeError("text should be str or list[str]")
