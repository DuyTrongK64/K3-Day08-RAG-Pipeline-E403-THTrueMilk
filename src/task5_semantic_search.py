"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from src.config import CHROMA_DISTANCE_METRIC, CHROMA_PERSIST_DIR


def semantic_search_with_dependencies(query: str, *, model, collection, top_k: int = 10) -> list[dict]:
    """Integration wrapper around the original Chroma query implementation."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    query_vector = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist) if CHROMA_DISTANCE_METRIC == "cosine" else 0.0
            output.append({
                "content": doc,
                "score": round(score, 4),
                "semantic_score": round(score, 4) if CHROMA_DISTANCE_METRIC == "cosine" else None,
                "raw_distance": float(dist),
                "score_type": "cosine_similarity" if CHROMA_DISTANCE_METRIC == "cosine" else "distance",
                "metadata": meta or {},
                "retrieval_source": "semantic",
            })
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not CHROMA_PERSIST_DIR.exists():
        return []
    from src.task4_chunking_indexing import get_collection, get_embedding_model

    return semantic_search_with_dependencies(
        query, model=get_embedding_model(), collection=get_collection(), top_k=top_k
    )


if __name__ == "__main__":
    # Test
    results = semantic_search("Quy định về thử việc", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
