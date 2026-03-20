from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import HTTPException

from src.api.schemas import RAGRequest
from src.clients.llm import get_llm, sanitize_llm_output
from src.rag.retriever import RerankRetriever


NO_ANSWER_RESPONSE = "对不起，基于现有知识库无法回答你的问题"
OUTPUT_PARSER = StrOutputParser()
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    你是一个智能助手，请基于以下上下文回答问题。如果上下文不足以回答，请说“对不起，基于现有知识库无法回答你的问题”。

    上下文：
    {context}

    问题：{question}
    回答：
    """
)


def _collect_sources(related_docs: list[Document]) -> list[str]:
    sources: list[str] = []
    for doc in related_docs:
        source = doc.metadata.get("source") or doc.metadata.get("file_path")
        if source and source not in sources:
            sources.append(source)
    return sources


def generate_rag_response(context: str, question: str) -> str:
    chain = RAG_PROMPT | get_llm() | OUTPUT_PARSER
    return sanitize_llm_output(chain.invoke({"context": context, "question": question}))


def handle_rag(request: RAGRequest) -> dict[str, str | list[str]]:
    try:
        related_docs = RerankRetriever().retrieve(request.query)
        if not related_docs:
            return {"response": NO_ANSWER_RESPONSE, "sources": []}

        context = "\n\n".join(doc.page_content for doc in related_docs)
        return {
            "response": generate_rag_response(context, request.query),
            "sources": _collect_sources(related_docs),
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
