from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core import config as cfg


TEXT_SEPARATORS = [
    "\n\n",
    "\r\n",
    "\n",
    " ",
    ".",
    "。",
    "；",
    "？",
    "！ ",
    ",",
    ";",
    "\u200b",
    "\uff0c",
    "\u3001",
    "\uff0e",
    "\u3002",
]

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
    separators=TEXT_SEPARATORS,
)


def split_text(text: str) -> list[str]:
    return TEXT_SPLITTER.split_text(text)


def split_file(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    if path.suffix.lower() != ".txt":
        return []
    return split_text(path.read_text(encoding="utf-8"))
