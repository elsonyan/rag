# -*- coding: utf-8 -*-
# @author: Elson Yan
# @file: routers.py
# @time: 2026/2/23 19:27
from src.services.rag_service import handle_rag
from src.services.chat_service import handle_chat

from src.routers.post_requests import RAGRequest, ChatRequest
from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def chat(request: ChatRequest):
    return await handle_chat(request)


@router.post("/rag")
async def rag(request: RAGRequest):
    return await handle_rag(request)
