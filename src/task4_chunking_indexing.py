"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
from pathlib import Path

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL as CONFIGURED_EMBEDDING_MODEL,
    STANDARDIZED_DIR,
)
from src.integration_adapters import parse_front_matter

CHROMA_DIR = CHROMA_PERSIST_DIR


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # Vì sao chọn 800? Theo yêu cầu của Lab Guide (Checkpoint 2), size 800 đủ lớn để chứa ngữ cảnh nhưng không vượt giới hạn token
CHUNK_OVERLAP = 100      # Vì sao chọn 100? Overlap 100 giúp bảo toàn ý nghĩa giữa các chunk khi bị cắt ngang
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = CONFIGURED_EMBEDDING_MODEL  # Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = CHROMA_COLLECTION_NAME

_embedding_model_instance = None
_chroma_client_instance = None

def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model_instance

def get_collection():
    import chromadb
    global _chroma_client_instance
    if _chroma_client_instance is None:
        _chroma_client_instance = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client_instance.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": CHROMA_DISTANCE_METRIC}
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents(standardized_dir: Path | None = None) -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    documents = []
    standardized_dir = standardized_dir or STANDARDIZED_DIR
    if not standardized_dir.exists():
        return documents
    for md_file in standardized_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        metadata = {"source": md_file.name, "type": doc_type}
        metadata.update(parse_front_matter(content))
        documents.append({
            "content": content,
            "metadata": metadata,
        })
    return documents


def chunk_documents(documents: list[dict], splitter=None) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    if splitter is None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            # Integration adapter: preserves recursive chunking when the optional package is absent.
            class _FallbackSplitter:
                def split_text(self, text):
                    step = CHUNK_SIZE - CHUNK_OVERLAP
                    return [text[start:start + CHUNK_SIZE] for start in range(0, len(text), step)]

            splitter = _FallbackSplitter()
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_id = hashlib.sha256(
                f"{doc['metadata'].get('source', '')}\x1f{i}\x1f{chunk_text}".encode("utf-8")
            ).hexdigest()
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i, "chunk_id": chunk_id}
            })
    return chunks


def embed_chunks(chunks: list[dict], model=None) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    if not chunks:
        return chunks
        
    model = model or get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict], collection=None):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # TODO: Implement indexing
    if not chunks:
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection = collection or get_collection()
    
    ids = [
        c["metadata"].get("chunk_id")
        or f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
