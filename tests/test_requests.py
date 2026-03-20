from pathlib import Path
import unittest

from pydantic import ValidationError

from src.api.schemas import ChatRequest, RAGRequest
from src.core import config as cfg


class RequestModelTests(unittest.TestCase):
    def test_chat_request_discards_blank_messages(self):
        request = ChatRequest(messages=["  你好  ", " ", "\n第二条\n"])
        self.assertEqual(request.messages, ["你好", "第二条"])

    def test_chat_request_rejects_all_blank_messages(self):
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[" ", "\n"])

    def test_rag_request_strips_query(self):
        request = RAGRequest(query="  投诉电话  ")
        self.assertEqual(request.query, "投诉电话")

    def test_rag_request_rejects_blank_query(self):
        with self.assertRaises(ValidationError):
            RAGRequest(query="   ")


class ConfigTests(unittest.TestCase):
    def test_storage_paths_are_absolute(self):
        self.assertTrue(Path(cfg.bloom_folder).is_absolute())
        self.assertTrue(Path(cfg.chroma_folder).is_absolute())
        self.assertTrue(Path(cfg.knowledge_file_folder).is_absolute())


if __name__ == "__main__":
    unittest.main()
