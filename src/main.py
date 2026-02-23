from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.routers import router

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
app.include_router(router, prefix="/chat", tags=["chat"])

# === 启动 ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
