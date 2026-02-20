from src.rag.text_spiltter import _text_splitter
from src.rag.text_vector import ChromaDB
from src.rag.file_detect import get_file_type
import os
from src.common import config as cfg

from src.rag.bloom_filter import BloomTextDedup

knowledge_file_folder = cfg.knowledge_file_folder

files = os.listdir(knowledge_file_folder)

dedup = BloomTextDedup()
text_vector = ChromaDB()
for file in files:
    embedding_vector = []
    file_path = os.path.join(knowledge_file_folder, file)
    filetype = get_file_type(file_path)
    if filetype == "txt":
        with open(file_path, "r", encoding='utf-8') as f:
            data = f.read()
        # 切词
        texts = _text_splitter(data)
        print(len(texts))
        # 生成词向量
        new_knowledge = [t.strip() for t in texts if not dedup.contains(t.strip())]
        # new_knowledge_vector = embedding_text(new_knowledge)
        if len(new_knowledge) > 0:
            print(len(new_knowledge))
            text_vector.add_texts(new_knowledge)
            dedup.add_batch(new_knowledge)
# 保存
dedup.save()

# result = text_vector.search_documents("投诉电话")
# [print(r) for r in result]
#
# print("-" * 100)
#
# result = text_vector.search_documents(["投诉电话", "货币兑换"])
# [print(r) for r in result]
