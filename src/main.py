from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.common import config as cfg
from src.rag.text_vector import ChromaDB
from src.common.retriever import ReRanker_Retriever
from src.rag.text_spiltter import _text_splitter
# LangChain 相关导入
from langchain_openai import ChatOpenAI

from src.services.llm_service import ChatRequest
from src.services.rag_service import RAGRequest

# --------------------------
# 1. 创建 LLM
# --------------------------
llm = ChatOpenAI(model=cfg.llm_model,
                 api_key=cfg.api_key,
                 base_url=cfg.llm_base_url
                 )
# --------------------------
# 2. 创建 FastAPI 应用
# --------------------------

app = FastAPI(title="LLM & RAG API")
# 允许跨域 (方便前端调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 1. 允许哪些域名访问
    allow_credentials=True,  # 2. 是否允许携带 Cookie/认证信息
    allow_methods=["*"],  # 3. 允许哪些 HTTP 方法 (GET, POST 等)
    allow_headers=["*"],  # 4. 允许哪些请求头
)


# === 常规聊天接口 ===
@app.post("/chat")
async def chat(request: ChatRequest):
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


# === RAG 接口 ===
@app.post("/rag")
async def rag(request: RAGRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is empty")
        # 检索相关文档
        reranker = ReRanker_Retriever()
        related_docs = reranker._get_relevant_documents(query=query)
        context = "\n\n".join([doc.page_content for doc in related_docs]) if related_docs else ""
        print("context", context)
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
        print(response)
        return {
            "response": response.split("</think>")[-1].strip(),
            "sources": [doc.metadata.get("source", "unknown") for doc in related_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 启动 ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
