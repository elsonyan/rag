from __future__ import annotations

from pathlib import Path

from src.core import config as cfg
from src.rag.bloom_filter import BloomTextDedup
from src.rag.splitter import split_file
from src.rag.vector_store import ChromaVectorStore


SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt"}


def build_metadata(file_path: Path, chunk_index: int) -> dict:
    return {
        "source": file_path.name,
        "file_path": str(file_path),
        "chunk_index": chunk_index,
        "file_type": file_path.suffix.lower().lstrip("."),
    }


def list_knowledge_files(knowledge_dir: Path | None = None) -> list[Path]:
    directory = knowledge_dir or Path(cfg.knowledge_file_folder)
    if not directory.exists():
        return []
    return sorted(file_path for file_path in directory.iterdir() if file_path.is_file())


def get_knowledge_overview(knowledge_dir: Path | None = None) -> dict:
    files = list_knowledge_files(knowledge_dir)
    supported_files = [file_path for file_path in files if file_path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES]
    return {
        "knowledge_dir": str((knowledge_dir or Path(cfg.knowledge_file_folder)).resolve()),
        "file_count": len(files),
        "supported_file_count": len(supported_files),
        "total_size_bytes": sum(file_path.stat().st_size for file_path in files),
        "files": [file_path.name for file_path in files],
    }


def ingest_knowledge_base(
        knowledge_dir: Path | None = None,
        dedup: BloomTextDedup | None = None,
        vector_store: ChromaVectorStore | None = None,
) -> list[dict]:
    directory = knowledge_dir or Path(cfg.knowledge_file_folder)
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge folder does not exist: {directory}")

    dedup = dedup or BloomTextDedup()
    vector_store = vector_store or ChromaVectorStore()
    ingest_summary: list[dict] = []

    for file_path in list_knowledge_files(directory):
        chunks = split_file(file_path)
        if not chunks:
            ingest_summary.append({"file": file_path.name, "status": "skipped", "reason": "unsupported file type"})
            continue

        new_chunks = []
        metadatas = []
        for chunk_index, chunk in enumerate(chunks):
            normalized_chunk = chunk.strip()
            if not normalized_chunk or dedup.contains(normalized_chunk):
                continue
            new_chunks.append(normalized_chunk)
            metadatas.append(build_metadata(file_path, chunk_index))

        if new_chunks:
            vector_store.add_texts(new_chunks, metadatas=metadatas)
            dedup.add_batch(new_chunks, auto_save=False)

        ingest_summary.append(
            {
                "file": file_path.name,
                "chunks": len(chunks),
                "new_chunks": len(new_chunks),
                "duplicates": len(chunks) - len(new_chunks),
            }
        )

    dedup.save()
    return ingest_summary
