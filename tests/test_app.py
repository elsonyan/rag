from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from src.app import app


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("knowledge_file_folder", payload)

    @patch("src.api.routes.handle_chat", return_value={"response": "你好"})
    def test_chat_endpoint(self, mock_handle_chat):
        response = self.client.post("/chat/", json={"messages": ["你好"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "你好"})

    @patch("src.api.routes.handle_rag", return_value={"response": "电话是34567890", "sources": ["service.txt"]})
    def test_rag_endpoint(self, mock_handle_rag):
        response = self.client.post("/chat/rag", json={"query": "电话是多少"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], ["service.txt"])

    def test_knowledge_overview(self):
        response = self.client.get("/knowledge/overview")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("file_count", payload)
        self.assertIn("files", payload)

    def test_invalid_chat_request_returns_422(self):
        response = self.client.post("/chat/", json={"messages": [" "]})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
