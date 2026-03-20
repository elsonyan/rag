import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    def __init__(
            self,
            model: str,
            api_key: str,
            base_url: str,
            timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY env or pass api_key.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
