"""Small shared configuration for integration boundaries."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


LANDING_DIR = _path("LANDING_DIR", "data/landing")
STANDARDIZED_DIR = _path("STANDARDIZED_DIR", "data/standardized")
CHROMA_PERSIST_DIR = _path("CHROMA_PERSIST_DIR", "chroma_db")
CHUNKS_PATH = _path("CHUNKS_PATH", "data/index/chunks.jsonl")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "labour_law_documents")
CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine").casefold()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

