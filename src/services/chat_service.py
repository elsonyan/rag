# -*- coding: utf-8 -*-
# @author: Elson Yan
# @file: chat_service.py
# @time: 2026/2/23 20:09
from fastapi import HTTPException

from src.routers.post_requests import ChatRequest
from src.services.llm_cilent import llm


async def handle_chat(request: ChatRequest):
    try:
        # 简单处理：取最后一条消息作为输入
        user_input = request.messages[-1] if request.messages else ""
        if not user_input.strip():
            raise HTTPException(status_code=400, detail="Empty input")
        # 直接调用 LLM（无上下文记忆）
        response = llm.invoke(user_input)
        return {"response": response.content.split("</think>")[-1].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
