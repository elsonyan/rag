from typing import Optional

from langchain_core.documents import Document

from src.common import config as cfg
from src.rag.text_vector import ChromaDB

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
import requests


class ReRanker_Retriever(BaseRetriever):
    similarity_search_k: int = Field(default=10, description="第一阶段检索返回的文档数量")
    re_ranker_k: int = Field(default=5, description="重排序后最终返回的文档数量")
    text_vector: Optional[ChromaDB] = Field(default=None, description="向量数据库实例")
    re_ranker_model: Optional[str] = Field(default=cfg.re_ranker_model, description="重排序模型名称")

    def __init__(self,
                 similarity_search_k: int = cfg.similarity_search_k,
                 re_ranker_k: int = cfg.re_ranker_k,
                 text_vector: ChromaDB = ChromaDB(),
                 re_ranker_model: str = cfg.re_ranker_model,
                 **kwargs
                 ) -> None:
        super().__init__(
            similarity_search_k=similarity_search_k,
            reranker_k=re_ranker_k,
            text_vector=text_vector,
            reranker_model=re_ranker_model,
            **kwargs
        )

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # get_vector
        initial_docs = self.text_vector.search_documents(text=query, k=self.similarity_search_k)
        if not initial_docs:
            return []
        return self._re_rank_documents(query=query, docs=initial_docs)

    def _re_rank_documents(self,
                           query: str,
                           docs: list[Document],
                           k=cfg.re_ranker_k):
        res = requests.post(url=cfg.re_ranker_url,
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
                            json={"model": self.re_ranker_model,
                                  "top_n": k,
                                  "query": query,
                                  "documents": [doc.page_content for doc in docs]
                                  }
                            )
        result = res.json()

        re_ranked_results = result.get("results", [])

        # 根据API返回的索引重新排序文档
        re_ranked_docs = []
        for res in re_ranked_results:
            index = res['index']
            # 把分数存到metadata中
            docs[index].metadata['rerank_score'] = res['relevance_score']
            docs[index].metadata['retrieve_rank'] = index  # 保存原始排名
            re_ranked_docs.append(docs[index])

        return re_ranked_docs


if __name__ == '__main__':
    re_ranker = ReRanker_Retriever()
    documents = re_ranker._get_relevant_documents(query="电话是多少？我没有你们的货币怎么办？")
    [print(doc, "\n-----") for doc in documents]
