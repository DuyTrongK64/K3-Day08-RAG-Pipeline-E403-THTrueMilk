"""Task 9: isolated hybrid retrieval, reranking, and PageIndex fallback.

Task 5 and 6 are expected to return candidate dictionaries but may be absent
while developed in parallel. ``retrieve_with_dependencies`` is the stable test
and integration boundary. Output always includes content, score, metadata, and
retrieval source fields while preserving unknown input fields.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any

try:
    from .task5_semantic_search import semantic_search
except ImportError:  # pragma: no cover - depends on parallel Task 5 work
    semantic_search = None

try:
    from .task6_lexical_search import lexical_search
except ImportError:  # pragma: no cover - depends on parallel Task 6 work
    lexical_search = None

from .task7_reranking import reciprocal_rank_fusion, rerank
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


class RetrievalPipelineError(RuntimeError):
    """Raised when no retriever can produce a usable result."""


def _validate(query: str, top_k: int, score_threshold: float) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
        raise TypeError("score_threshold must be numeric")
    if not math.isfinite(float(score_threshold)):
        raise ValueError("score_threshold must be finite")


def _score(candidate: Mapping[str, Any]) -> float | None:
    for key in ("score", "similarity", "relevance_score"):
        if candidate.get(key) is not None:
            try:
                value = float(candidate[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    if candidate.get("distance") is not None:
        try:
            distance = float(candidate["distance"])
        except (TypeError, ValueError):
            return None
        if math.isfinite(distance) and 0.0 <= distance <= 2.0:
            return 1.0 - distance
    return None


def _normalize(candidate: Mapping[str, Any], source: str) -> dict[str, Any] | None:
    content = next(
        (
            candidate.get(key)
            for key in ("content", "text", "document", "page_content", "body")
            if candidate.get(key)
        ),
        None,
    )
    if not isinstance(content, str) or not content.strip():
        return None
    metadata = candidate.get("metadata")
    result = dict(candidate)
    result["content"] = content.strip()
    result["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    numeric_score = _score(candidate)
    result["score"] = numeric_score if numeric_score is not None else 0.0
    result["retrieval_source"] = source
    result["source"] = source
    # Integration adapter: preserves calibrated scores from the original retrievers.
    if source == "semantic":
        explicit_semantic = candidate.get("semantic_score")
        result["semantic_score"] = (
            _score({"score": explicit_semantic}) if explicit_semantic is not None else None
        )
        if result["semantic_score"] is None and candidate.get("score_type") in (None, "cosine_similarity"):
            result["semantic_score"] = numeric_score
    elif source == "lexical":
        result["lexical_score"] = candidate.get("lexical_score", numeric_score)
    else:
        result[f"{source}_score"] = numeric_score
    return result


def _identity(candidate: Mapping[str, Any]) -> str:
    metadata = candidate.get("metadata")
    meta = metadata if isinstance(metadata, Mapping) else {}
    for key in ("chunk_id", "id"):
        if meta.get(key) not in (None, ""):
            return f"{key}:{meta[key]}"
    if meta.get("document_id") not in (None, ""):
        return f"document:{meta['document_id']}:{meta.get('page', '')}:{meta.get('section', '')}"
    canonical = "\x1f".join(
        (
            str(meta.get("source") or meta.get("url") or meta.get("path") or ""),
            str(meta.get("page") or ""),
            " ".join(str(candidate.get("content", "")).split()).casefold(),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_search(
    name: str,
    function: Callable[..., list[dict]] | None,
    query: str,
    top_k: int,
    errors: dict[str, str],
) -> list[dict]:
    if function is None:
        errors[name] = "dependency is unavailable"
        return []
    try:
        raw = function(query, top_k=top_k)
        if not isinstance(raw, list):
            raise TypeError("returned a non-list result")
        return raw
    except Exception as exc:
        # Do not propagate provider messages: an SDK exception can contain credentials.
        errors[name] = f"{type(exc).__name__}: dependency failed"
        return []


def _fuse(semantic: list[dict], lexical: list[dict], top_k: int) -> list[dict]:
    fused = reciprocal_rank_fusion([semantic, lexical], top_k=top_k)
    details: dict[str, dict[str, Any]] = {}
    membership: dict[str, set[str]] = {}
    for source, ranked in (("semantic", semantic), ("lexical", lexical)):
        for item in ranked:
            key = _identity(item)
            membership.setdefault(key, set()).add(source)
            details.setdefault(key, {}).update({f"{source}_score": item.get(f"{source}_score")})
    for item in fused:
        key = _identity(item)
        item.update(details.get(key, {}))
        sources = membership.get(key, set())
        retrieval_source = "hybrid" if len(sources) > 1 else next(iter(sources), "hybrid")
        item["retrieval_source"] = retrieval_source
        item["source"] = retrieval_source
    return fused


def _combine_fallback(
    query: str,
    hybrid: list[dict],
    pageindex: list[dict],
    rerank_fn: Callable[..., list[dict]],
    top_k: int,
) -> list[dict]:
    if not hybrid:
        return pageindex[:top_k]
    combined = reciprocal_rank_fusion([pageindex, hybrid], top_k=max(top_k * 2, 10))
    for item in combined:
        if item.get("retrieval_source") not in {"pageindex", "semantic", "lexical", "hybrid"}:
            item["retrieval_source"] = "hybrid"
        item["source"] = item["retrieval_source"]
    return rerank_fn(query, combined, top_k=top_k)


# Integration contract: dependencies use only (query, top_k) and normalized dicts.
def retrieve_with_dependencies(
    query: str,
    *,
    semantic_search_fn: Callable[..., list[dict]] | None,
    lexical_search_fn: Callable[..., list[dict]] | None,
    rerank_fn: Callable[..., list[dict]],
    pageindex_search_fn: Callable[..., list[dict]],
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Run the retrieval pipeline with injectable, independently failing adapters."""
    _validate(query, top_k, score_threshold)
    candidate_k = max(top_k * 3, 10)
    errors: dict[str, str] = {}
    raw_semantic = _safe_search("semantic", semantic_search_fn, query, candidate_k, errors)
    raw_lexical = _safe_search("lexical", lexical_search_fn, query, candidate_k, errors)
    semantic = [item for raw in raw_semantic if isinstance(raw, Mapping) and (item := _normalize(raw, "semantic"))]
    lexical = [item for raw in raw_lexical if isinstance(raw, Mapping) and (item := _normalize(raw, "lexical"))]

    # Threshold is deliberately based only on original Task 5 scores, never RRF.
    semantic_scores = [item["semantic_score"] for item in semantic if item.get("semantic_score") is not None]
    best_semantic_score = max(semantic_scores) if semantic_scores else None
    fused = _fuse(semantic, lexical, max(top_k * 2, 10))
    try:
        hybrid = rerank_fn(query, fused, top_k=top_k) if fused else []
    except Exception as exc:
        errors["rerank"] = f"{type(exc).__name__}: dependency failed"
        hybrid = fused[:top_k]

    should_fallback = not fused or best_semantic_score is None or best_semantic_score < float(score_threshold)
    page_results: list[dict] = []
    if should_fallback:
        raw_page = _safe_search("pageindex", pageindex_search_fn, query, candidate_k, errors)
        page_results = [
            item for raw in raw_page if isinstance(raw, Mapping) and (item := _normalize(raw, "pageindex"))
        ]
        if page_results:
            try:
                return _combine_fallback(query, hybrid, page_results, rerank_fn, top_k)
            except Exception as exc:
                errors["final_rerank"] = f"{type(exc).__name__}: dependency failed"
                return page_results[:top_k] if not hybrid else hybrid[:top_k]

    if hybrid:
        return hybrid[:top_k]
    if page_results:
        return page_results[:top_k]
    if {"semantic", "lexical", "pageindex"}.issubset(errors):
        summary = "; ".join(f"{name}={message}" for name, message in errors.items())
        raise RetrievalPipelineError(f"No retrieval dependency produced results: {summary}")
    return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Run hybrid search, RRF fusion, reranking, and conditional PageIndex fallback."""
    return retrieve_with_dependencies(
        query,
        semantic_search_fn=semantic_search,
        lexical_search_fn=lexical_search,
        rerank_fn=rerank,
        pageindex_search_fn=pageindex_search,
        top_k=top_k,
        score_threshold=score_threshold,
    )
