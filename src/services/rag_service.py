# -*- coding: utf-8 -*-
# @author: Elson Yan
# @file: rag_service.py
# @time: 2026/2/23 20:10
from fastapi import HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.common.retriever import ReRanker_Retriever
from src.routers.post_requests import RAGRequest
from src.services.llm_cilent import llm


async def handle_rag(request: RAGRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is empty")
        # 检索相关文档
        re_ranker = ReRanker_Retriever()
        related_docs = re_ranker._get_relevant_documents(query=query)
        context = "\n\n".join([doc.page_content for doc in related_docs]) if related_docs else ""
        # 构建 Prompt
        prompt_template = """
        你是一个智能助手，请基于以下上下文回答问题。如果上下文不足以回答，请说“对不起，基于现有知识库无法回答你的问题”。
    
        上下文：
        {context}
    
        问题：{question}
        回答：
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"context": context, "question": query})
        return {
            "response": response.split("</think>")[-1].strip(),
            "sources": [doc.metadata.get("source", "unknown") for doc in related_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
