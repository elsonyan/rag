from __future__ import annotations

from typing import Optional

import requests
from langchain_core.documents import Document

from src.core import config as cfg
from src.rag.vector_store import ChromaVectorStore


class RerankRetriever:
    def __init__(
            self,
            similarity_search_k: int = cfg.similarity_search_k,
            re_ranker_k: int = cfg.re_ranker_k,
            vector_store: Optional[ChromaVectorStore] = None,
            re_ranker_model: str = cfg.re_ranker_model,
            request_timeout: float = 30.0,
    ) -> None:
        self.similarity_search_k = similarity_search_k
        self.re_ranker_k = re_ranker_k
        self.vector_store = vector_store or ChromaVectorStore()
        self.re_ranker_model = re_ranker_model
        self.request_timeout = request_timeout

    def retrieve(self, query: str) -> list[Document]:
        initial_docs = self.vector_store.search_documents(query, k=self.similarity_search_k)
        if not initial_docs:
            return []
        return self._re_rank_documents(query=query, docs=initial_docs)

    def _re_rank_documents(self, query: str, docs: list[Document]) -> list[Document]:
        if not cfg.api_key:
            return docs[:self.re_ranker_k]

        try:
            response = requests.post(
                url=cfg.re_ranker_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.api_key}",
                },
                json={
                    "model": self.re_ranker_model,
                    "top_n": self.re_ranker_k,
                    "query": query,
                    "documents": [doc.page_content for doc in docs],
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException:
            return docs[:self.re_ranker_k]

        re_ranked_results = result.get("results", [])
        re_ranked_docs: list[Document] = []
        for rank_index, item in enumerate(re_ranked_results):
            index = item.get("index")
            if index is None or index >= len(docs):
                continue
            docs[index].metadata["rerank_score"] = item.get("relevance_score")
            docs[index].metadata["retrieve_rank"] = index
            docs[index].metadata["rerank_rank"] = rank_index
            re_ranked_docs.append(docs[index])

        return re_ranked_docs or docs[:self.re_ranker_k]
