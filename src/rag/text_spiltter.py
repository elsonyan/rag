from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from src.rag.bloom_filter import BloomTextDedup

from src.common import config as cfg
from src.rag.file_detect import get_file_type


def _text_splitter(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size,
                                                   chunk_overlap=cfg.chunk_overlap,
                                                   separators=[
                                                       "\n\n",
                                                       "\r\n",
                                                       "\n",
                                                       " ",
                                                       ".",
                                                       "。",
                                                       "；",
                                                       "？",
                                                       "！ ",
                                                       ",",
                                                       ".",
                                                       ";",
                                                       "\u200b",  # Zero-width space
                                                       "\uff0c",  # Fullwidth comma
                                                       "\u3001",  # Ideographic comma
                                                       "\uff0e",  # Fullwidth full stop
                                                       "\u3002",  # Ideographic full stop
                                                   ]
                                                   )
    texts = text_splitter.split_text(text)
    return texts


def _llm_splitter(text, llm) -> str:
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的文档处理助手。请将以下文本切分成语义完整的段落，直接展示，不需要多余的话
    每个段落 300-500 字，保持主题一致性。

    要求：
    1. 不要在句子中间切断
    2. 每个段落应有完整的主谓宾
    3. 用 <CHUNK> 标签分隔

    待处理文本：
    {text}

    切分结果：
    """)

    chain = prompt | llm

    def llm_based_split(text):
        result = chain.invoke({"text": text})
        # 解析 <CHUNK> 标签
        chunks = [c.strip() for c in result.content.split("<CHUNK>") if c.strip()]
        return chunks

    return llm_based_split(text)


def _read_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def splitter(file_name: str,
             file_type=None
             ):
    file_type = file_type if file_type else get_file_type(file_name)
    match file_type:
        case "txt":
            return _text_splitter(_read_text(file_name))

        case _:
            return None


def text_save(text: Optional[str | list[str]], bloom_conn: BloomTextDedup):
    try:
        if isinstance(text, str):
            bloom_conn.add(text)
        if isinstance(text, list):
            bloom_conn.add_batch(text)
    except Exception as e:
        print(e)
    finally:
        bloom_conn.save()


if __name__ == '__main__':
    dedup = BloomTextDedup()

    text_save("123", dedup)
