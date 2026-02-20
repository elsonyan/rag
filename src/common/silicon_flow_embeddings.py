import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    def __init__(
            self,
            model: str,
            api_key: str,
            base_url: str
    ):
        self.model = model
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY env or pass api_key.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        response = requests.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


if __name__ == '__main__':
    from src.common import config as cfg

    embedding = SiliconFlowEmbeddings(model=cfg.silicon_embedding_model,
                                      api_key=cfg.api_key)
    print(embedding.embed_query("nihao"))
