# MicroAgents RAG

一个基于 FastAPI、Chroma 和 SiliconFlow 的轻量 RAG 服务。

## 目录

- `src/api`: HTTP 路由和请求/响应模型
- `src/core`: 配置与路径
- `src/clients`: 外部模型与 Embedding 客户端
- `src/rag`: 向量检索、切分和去重
- `src/services`: 业务编排
- `scripts`: 导库和本地调试脚本
- `tests`: 单元测试
- `data`: 知识文件与本地持久化数据

## 环境准备

```bash
uv venv --python 3.12 .venv
uv sync
cp .env.example .env
```

至少需要配置：

```env
SILICONFLOW_API_KEY=your_api_key_here
```

## 启动服务

```bash
uv run uvicorn src.main:app --reload
```

服务启动后可访问：

- `GET /health`
- `POST /chat/`
- `POST /chat/rag`
- `GET /knowledge/overview`

## 导入知识库

将 `.txt` 文件放到 `data/knowledge_files/`，然后执行：

```bash
uv run python scripts/ingest_knowledge.py
```

## 本地调试请求

```bash
uv run python scripts/demo_request.py --mode rag --message "电话是多少？"
uv run python scripts/demo_request.py --mode chat --message "你好"
```

## 运行测试

```bash
uv run python -m unittest discover -s tests
```
