from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.rag.embedding import _embedding
from src.common import config as cfg
from hashlib import sha256


class ChromaDB:
    def __init__(self,
                 vector_store: VectorStore = Chroma(collection_name="knowledge_vector",
                                                    embedding_function=_embedding,
                                                    persist_directory=cfg.chroma_folder),
                 ):
        self.vector_store = vector_store

    def add_documents(self, text: Optional[str | List[str]]):
        if isinstance(text, str):
            self.vector_store.add_documents(documents=[text], ids=[sha256(text.encode()).hexdigest()])
        if isinstance(text, list):
            self.vector_store.add_documents(documents=text, ids=[sha256(t.encode()).hexdigest() for t in text])

    def search_documents(self, text: Optional[str | List[str]], k: int = cfg.similarity_search_k) -> List[Document]:
        if isinstance(text, str):
            return self.vector_store.similarity_search(query=text, k=k)
        if isinstance(text, list):
            result = []
            for t in text:
                for x in self.vector_store.similarity_search(query=t, k=k):
                    result.append(x)
            return result
        return None

    def add_texts(self, text: Optional[str | List[str]]):
        if isinstance(text, str):
            self.vector_store.add_texts(texts=[text], ids=[sha256(text.encode()).hexdigest()])
        if isinstance(text, list):
            self.vector_store.add_texts(texts=text, ids=[sha256(t.encode()).hexdigest() for t in text])
