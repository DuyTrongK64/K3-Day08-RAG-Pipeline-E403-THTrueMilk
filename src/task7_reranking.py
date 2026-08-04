"""Task 7: pluggable reranking for normalized retrieval candidates.

Public APIs are :func:`rerank` and :func:`reciprocal_rank_fusion`. Candidates
may use ``content`` (or a common text alias), a numeric score, and metadata.
Unknown fields are preserved. ``RERANK_BACKEND`` selects ``auto``, ``jina``,
or deterministic offline ``rrf``; Jina configuration is read lazily from the
environment and failures fall back offline.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from typing import Any


class RerankError(RuntimeError):
    """Raised for invalid reranker responses or configuration."""


_CONTENT_ALIASES = ("content", "text", "document", "page_content", "body")
_SCORE_ALIASES = ("score", "similarity", "relevance_score")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _validate(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_candidate(candidate: Mapping[str, Any], rank: int) -> dict[str, Any] | None:
    content = next((candidate.get(k) for k in _CONTENT_ALIASES if candidate.get(k)), None)
    if not isinstance(content, str) or not content.strip():
        return None
    output = dict(candidate)
    metadata = candidate.get("metadata")
    output["content"] = content.strip()
    output["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    score = next((candidate.get(k) for k in _SCORE_ALIASES if candidate.get(k) is not None), None)
    if score is None and candidate.get("distance") is not None:
        distance = _as_float(candidate.get("distance"), default=math.inf)
        score = 1.0 - distance if 0.0 <= distance <= 2.0 else 0.0
    output["score"] = _as_float(score)
    output.setdefault("original_rank", rank)
    source = output.get("retrieval_source") or output.get("source")
    if not source:
        source = output["metadata"].get("retrieval_source") or output["metadata"].get("source")
    if source:
        output["retrieval_source"] = str(source)
    return output


def _document_identity(candidate: Mapping[str, Any]) -> str:
    metadata = candidate.get("metadata")
    meta = metadata if isinstance(metadata, Mapping) else {}
    for key in ("chunk_id", "id"):
        if meta.get(key) not in (None, ""):
            return f"{key}:{meta[key]}"
    if meta.get("document_id") not in (None, ""):
        return f"document:{meta['document_id']}:{meta.get('page', '')}:{meta.get('section', '')}"
    content = str(candidate.get("content", "")).strip()
    source = meta.get("source") or meta.get("url") or meta.get("path") or ""
    page = meta.get("page", "")
    identity = f"{source}\x1f{page}\x1f{' '.join(content.split()).casefold()}"
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], top_k: int = 5, rrf_k: int = 60
) -> list[dict]:
    """Fuse ranked lists using stable document identity and RRF ordering."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    if not isinstance(ranked_lists, list):
        raise TypeError("ranked_lists must be a list")

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranked in ranked_lists:
        if not isinstance(ranked, list):
            continue
        seen_in_list: set[str] = set()
        for rank, raw in enumerate(ranked, 1):
            if not isinstance(raw, Mapping):
                continue
            item = _normalize_candidate(raw, rank)
            if item is None:
                continue
            identity = _document_identity(item)
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + rank)
            if identity not in documents:
                documents[identity] = item
                first_seen[identity] = seen_counter
                seen_counter += 1

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    results: list[dict[str, Any]] = []
    for identity in ordered[:top_k]:
        item = dict(documents[identity])
        item["metadata"] = dict(documents[identity]["metadata"])
        item["fusion_score"] = scores[identity]
        results.append(item)
    return results


# Backward-compatible name used by the starter README.
def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Alias for :func:`reciprocal_rank_fusion`."""
    return reciprocal_rank_fusion(ranked_lists, top_k=top_k, rrf_k=k)


def _offline_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    query_tokens = set(_TOKEN_RE.findall(query.casefold()))
    scored: list[dict[str, Any]] = []
    for item in candidates:
        document_tokens = set(_TOKEN_RE.findall(item["content"].casefold()))
        overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
        phrase_bonus = 0.15 if query.casefold() in item["content"].casefold() else 0.0
        rank_bonus = 1.0 / (60 + int(item["original_rank"]))
        result = dict(item)
        result["metadata"] = dict(item["metadata"])
        result["rerank_score"] = overlap + phrase_bonus + rank_bonus
        scored.append(result)
    scored.sort(key=lambda item: (-item["rerank_score"], item["original_rank"]))
    return scored[:top_k]


def _request_jina(query: str, candidates: list[dict], top_k: int) -> object:
    """Call Jina lazily; kept separate so tests can monkeypatch the transport."""
    import requests

    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        raise RerankError("JINA_API_KEY is required for the jina backend")
    try:
        timeout = float(os.getenv("RERANK_TIMEOUT_SECONDS", "20"))
    except ValueError as exc:
        raise RerankError("RERANK_TIMEOUT_SECONDS must be numeric") from exc
    response = requests.post(
        os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
            "query": query,
            "documents": [item["content"] for item in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _normalize_jina_response(raw: object, candidates: list[dict], top_k: int) -> list[dict]:
    payload = raw if isinstance(raw, Mapping) else {}
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise RerankError("Jina response is missing a results list")
    output: list[dict[str, Any]] = []
    used: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidates):
            raise RerankError("Jina returned an invalid candidate index")
        if index in used:
            continue
        score = row.get("relevance_score", row.get("score"))
        if score is None:
            raise RerankError("Jina result is missing relevance_score")
        item = dict(candidates[index])
        item["metadata"] = dict(candidates[index]["metadata"])
        item["rerank_score"] = _as_float(score)
        output.append(item)
        used.add(index)
    output.sort(key=lambda item: (-item["rerank_score"], item["original_rank"]))
    return output[:top_k]


# Integration contract: Task 5/6 candidates enter here without in-place mutation.
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score and re-order candidates based on relevance to query."""
    _validate(query, top_k)
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    normalized = [
        item
        for rank, raw in enumerate(candidates, 1)
        if isinstance(raw, Mapping) and (item := _normalize_candidate(raw, rank)) is not None
    ]
    if not normalized:
        return []
    backend = os.getenv("RERANK_BACKEND", "auto").strip().casefold()
    if backend not in {"auto", "jina", "rrf"}:
        raise ValueError("RERANK_BACKEND must be auto, jina, or rrf")
    use_jina = backend == "jina" or (backend == "auto" and bool(os.getenv("JINA_API_KEY", "").strip()))
    if use_jina:
        try:
            return _normalize_jina_response(_request_jina(query, normalized, top_k), normalized, top_k)
        except Exception:
            # External reranking is optional; deterministic fallback keeps retrieval usable.
            pass
    return _offline_rerank(query, normalized, min(top_k, len(normalized)))
