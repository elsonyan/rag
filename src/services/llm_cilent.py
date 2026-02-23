# -*- coding: utf-8 -*-
# @author: Elson Yan
# @file: llm_cilent.py
# @time: 2026/2/23 19:30

from langchain_openai import ChatOpenAI
from src.common import config as cfg

llm = ChatOpenAI(model=cfg.llm_model,
                 api_key=cfg.api_key,
                 base_url=cfg.llm_base_url
                 )
