# -*- coding: utf-8 -*-
# @author: Elson Yan
# @file: post_requests.py
# @time: 2026/2/23 20:11
from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: List[str]  # 简化：只传用户最新消息，或可扩展为 [{"role": "user", "content": "..."}]
    session_id: Optional[str] = None


class RAGRequest(BaseModel):
    query: str
    session_id: Optional[str] = None