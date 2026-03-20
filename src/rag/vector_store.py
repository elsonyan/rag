from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.core import config as cfg
from src.rag.embedding import get_embedding_proxy


class ChromaVectorStore:
    def __init__(self, vector_store: VectorStore | None = None):
        cfg.ensure_directories()
        self.vector_store = vector_store or Chroma(
            collection_name="knowledge_vector",
            embedding_function=get_embedding_proxy(),
            persist_directory=cfg.chroma_folder,
        )

    @staticmethod
    def _build_id(text: str, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        source = metadata.get("source", "")
        chunk_index = metadata.get("chunk_index", "")
        return sha256(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()

    @classmethod
    def _deduplicate_documents(cls, documents: Iterable[Document]) -> list[Document]:
        deduplicated_docs: list[Document] = []
        seen_keys: set[str] = set()
        for document in documents:
            key = cls._build_id(document.page_content, document.metadata)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated_docs.append(document)
        return deduplicated_docs

    def search_documents(self, text: str | Sequence[str] | None, k: int = cfg.similarity_search_k) -> list[Document]:
        if isinstance(text, str):
            normalized_text = text.strip()
            if not normalized_text:
                return []
            return self.vector_store.similarity_search(query=normalized_text, k=k)

        if isinstance(text, Sequence):
            result: list[Document] = []
            for query in text:
                normalized_query = query.strip()
                if not normalized_query:
                    continue
                result.extend(self.vector_store.similarity_search(query=normalized_query, k=k))
            return self._deduplicate_documents(result)

        return []

    def add_texts(
            self,
            text: str | Sequence[str] | None,
            metadatas: Sequence[dict] | None = None,
    ) -> list[str]:
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, Sequence):
            texts = list(text)
        else:
            return []

        if metadatas is None:
            pairs = [(item, {}) for item in texts]
        else:
            normalized_metadatas = list(metadatas)
            if len(normalized_metadatas) != len(texts):
                raise ValueError("metadatas length must match texts length")
            pairs = list(zip(texts, normalized_metadatas))

        cleaned_pairs = [
            (item.strip(), metadata)
            for item, metadata in pairs
            if item and item.strip()
        ]
        if not cleaned_pairs:
            return []

        cleaned_texts = [item for item, _ in cleaned_pairs]
        normalized_metadatas = [metadata for _, metadata in cleaned_pairs]
        ids = [
            self._build_id(text_item, metadata)
            for text_item, metadata in zip(cleaned_texts, normalized_metadatas)
        ]
        self.vector_store.add_texts(texts=cleaned_texts, metadatas=normalized_metadatas, ids=ids)
        return ids
