from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.core import config as cfg


def sanitize_llm_output(content: str) -> str:
    return content.split("</think>")[-1].strip()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    if not cfg.api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not configured.")

    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.api_key,
        base_url=cfg.llm_base_url,
        max_retries=1,
        timeout=30,
    )
