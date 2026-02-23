import os

api_key = "sk-odrdsvtdzhnsnqtwapxfrnjsbgnnopagutmmwimvsxljktbm" # expired
llm_base_url = "https://api.siliconflow.cn/v1"
llm_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
silicon_embedding_base_url = "https://api.siliconflow.cn/v1/embeddings"
silicon_embedding_model = "netease-youdao/bce-embedding-base_v1"
embedding_model = "text-embedding-v4"
chunk_size = 100
chunk_overlap = 10
similarity_search_k = 30
re_ranker_model = "netease-youdao/bce-reranker-base_v1"
re_ranker_url = "https://api.siliconflow.cn/v1/rerank"
re_ranker_k = 3

src_dir = r"F:\rag\src"
bloom_folder = os.path.join(src_dir, r"storage\bloom")
chroma_folder = os.path.join(src_dir, r"storage\chroma")
knowledge_file_folder = os.path.join(src_dir, r"knowledge_files")
