import argparse
from pathlib import Path

from src.core import config as cfg
from src.services.knowledge import ingest_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local knowledge files into the vector store.")
    parser.add_argument(
        "--knowledge-dir",
        default=cfg.knowledge_file_folder,
        help="Directory that contains source knowledge files.",
    )
    args = parser.parse_args()

    for item in ingest_knowledge_base(Path(args.knowledge_dir)):
        print(item)


if __name__ == "__main__":
    main()
