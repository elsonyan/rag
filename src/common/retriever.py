from typing import Any, Optional

from langchain_core.documents import Document

from src.common import config as cfg
from src.rag.text_vector import ChromaDB

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
import requests


class ReRanker_Retriever(BaseRetriever):
    similarity_search_k: int = Field(default=10, description="第一阶段检索返回的文档数量")
    reranker_k: int = Field(default=5, description="重排序后最终返回的文档数量")
    text_vector: Optional[ChromaDB] = Field(default=None, description="向量数据库实例")
    reranker_model: Optional[str] = Field(default="BAAI/bge-reranker-base", description="重排序模型名称")

    def __init__(self,
                 similarity_search_k: int = cfg.similarity_search_k,
                 reranker_k: int = cfg.reranker_k,
                 text_vector: ChromaDB = ChromaDB(),
                 reranker_model: str = cfg.reranker_model,
                 **kwargs
                 ) -> None:
        super().__init__(
            similarity_search_k=similarity_search_k,
            reranker_k=reranker_k,
            text_vector=text_vector,
            reranker_model=reranker_model,
            **kwargs
        )

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # get_vector
        initial_docs = self.text_vector.search_documents(text=query, k=self.similarity_search_k)
        if not initial_docs:
            return []
        return self._rerank_documents(query=query, docs=initial_docs)

    def _rerank_documents(self,
                          query: str,
                          docs: list[Document],
                          k=cfg.reranker_k):
        res = requests.post(url=cfg.reranker_url,
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
                            json={"model": self.reranker_model,
                                  "top_n": k,
                                  "query": query,
                                  "documents": [doc.page_content for doc in docs]
                                  }
                            )
        result = res.json()

        reranked_results = result.get("results", [])

        # 根据API返回的索引重新排序文档
        reranked_docs = []
        for res in reranked_results:
            index = res['index']
            # 把分数存到metadata中
            docs[index].metadata['rerank_score'] = res['relevance_score']
            docs[index].metadata['retrieve_rank'] = index  # 保存原始排名
            reranked_docs.append(docs[index])

        return reranked_docs


if __name__ == '__main__':
    reranker = ReRanker_Retriever()
    documents = reranker._get_relevant_documents("请问投诉电话是多少")
    [print(doc, "\n-----") for doc in documents]
