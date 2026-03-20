from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from fastapi import HTTPException
from langchain_core.documents import Document

from src.api.schemas import ChatRequest, RAGRequest
from src.services.chat import handle_chat
from src.services.knowledge import build_metadata, get_knowledge_overview
from src.services.rag import handle_rag


class ChatServiceTests(unittest.TestCase):
    @patch("src.services.chat.get_llm")
    def test_handle_chat_returns_sanitized_content(self, mock_get_llm):
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="<think>hidden</think>hello")
        mock_get_llm.return_value = mock_llm

        result = handle_chat(ChatRequest(messages=["你好"]))

        self.assertEqual(result, {"response": "hello"})

    @patch("src.services.chat.get_llm", side_effect=RuntimeError("missing api key"))
    def test_handle_chat_maps_runtime_error_to_503(self, mock_get_llm):
        with self.assertRaises(HTTPException) as context:
            handle_chat(ChatRequest(messages=["你好"]))

        self.assertEqual(context.exception.status_code, 503)


class RagServiceTests(unittest.TestCase):
    @patch("src.services.rag.RerankRetriever")
    def test_handle_rag_returns_default_message_when_no_docs(self, mock_retriever_cls):
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        result = handle_rag(RAGRequest(query="电话是多少"))

        self.assertEqual(result["sources"], [])
        self.assertIn("无法回答", result["response"])

    @patch("src.services.rag.generate_rag_response", return_value="电话是34567890")
    @patch("src.services.rag.RerankRetriever")
    def test_handle_rag_returns_sources(self, mock_retriever_cls, mock_generate_rag_response):
        docs = [
            Document(page_content="电话是34567890", metadata={"source": "service.txt"}),
            Document(page_content="接受知识交换", metadata={"source": "service.txt"}),
        ]
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = docs
        mock_retriever_cls.return_value = mock_retriever

        result = handle_rag(RAGRequest(query="电话是多少"))

        self.assertEqual(result["response"], "电话是34567890")
        self.assertEqual(result["sources"], ["service.txt"])


class KnowledgeServiceTests(unittest.TestCase):
    def test_build_metadata_contains_expected_fields(self):
        metadata = build_metadata(Path("service.txt"), 3)
        self.assertEqual(metadata["source"], "service.txt")
        self.assertEqual(metadata["chunk_index"], 3)
        self.assertEqual(metadata["file_type"], "txt")

    def test_get_knowledge_overview_shape(self):
        overview = get_knowledge_overview()
        self.assertIn("knowledge_dir", overview)
        self.assertIn("file_count", overview)
        self.assertIn("files", overview)


if __name__ == "__main__":
    unittest.main()
