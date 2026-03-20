from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.services.knowledge import ingest_knowledge_base


class FakeDedup:
    def __init__(self):
        self.saved = False
        self.items: set[str] = set()

    def contains(self, text: str) -> bool:
        return text in self.items

    def add_batch(self, texts: list[str], auto_save: bool = False) -> dict:
        self.items.update(texts)
        return {"total": len(texts), "new": len(texts), "duplicates": 0}

    def save(self) -> None:
        self.saved = True


class FakeVectorStore:
    def __init__(self):
        self.calls: list[tuple[list[str], list[dict]]] = []

    def add_texts(self, texts: list[str], metadatas: list[dict]) -> list[str]:
        self.calls.append((texts, metadatas))
        return [str(index) for index, _ in enumerate(texts)]


class KnowledgeIngestTests(unittest.TestCase):
    def test_ingest_knowledge_base_reads_txt_files(self):
        with TemporaryDirectory() as temp_dir:
            knowledge_dir = Path(temp_dir)
            (knowledge_dir / "service.txt").write_text("第一段。\n第二段。", encoding="utf-8")
            (knowledge_dir / "ignore.md").write_text("# heading", encoding="utf-8")

            dedup = FakeDedup()
            vector_store = FakeVectorStore()
            summary = ingest_knowledge_base(knowledge_dir, dedup=dedup, vector_store=vector_store)

            self.assertEqual(len(summary), 2)
            self.assertTrue(any(item["file"] == "service.txt" for item in summary))
            self.assertTrue(any(item["file"] == "ignore.md" and item["status"] == "skipped" for item in summary))
            self.assertTrue(vector_store.calls)
            self.assertTrue(dedup.saved)


if __name__ == "__main__":
    unittest.main()
