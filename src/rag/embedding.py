from typing import Optional, List
from src.common.silicon_flow_embeddings import SiliconFlowEmbeddings
from src.common import config as cfg

_embedding = SiliconFlowEmbeddings(
    model=cfg.silicon_embedding_model,
    api_key=cfg.api_key,
    base_url=cfg.silicon_embedding_base_url
)


def embedding_text(text: Optional[str | List[str]]) -> List[List[float]]:
    try:
        if isinstance(text, str):
            return [_embedding.embed_query(text)]
        if isinstance(text, list):
            return _embedding.embed_documents(text)
        raise TypeError("text should be str or list[str]")
    except Exception as e:
        print(e)
