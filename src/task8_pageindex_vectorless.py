"""Task 8: lazy PageIndex vectorless retrieval adapter.

``pageindex_search`` returns normalized candidates with rank-derived scores
explicitly labelled ``score_type=rank``. Environment variables are
``PAGEINDEX_API_KEY``, ``PAGEINDEX_INDEX_ID``, ``PAGEINDEX_BASE_URL``, and
``PAGEINDEX_TIMEOUT_SECONDS``. Missing configuration/SDK raises
``PageIndexUnavailableError`` for Task 9 to handle safely.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PageIndexError(RuntimeError):
    """Base PageIndex adapter error."""


class PageIndexUnavailableError(PageIndexError):
    """PageIndex is not installed or configured."""


def _extract_rows(raw_results: object) -> list[object]:
    if isinstance(raw_results, list):
        return raw_results
    if not isinstance(raw_results, Mapping):
        return []
    for key in ("results", "data", "matches", "items"):
        value = raw_results.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            nested = _extract_rows(value)
            if nested:
                return nested
    nodes = raw_results.get("retrieved_nodes")
    if isinstance(nodes, list):
        flattened: list[object] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            groups = node.get("relevant_contents", [])
            if isinstance(groups, list):
                for group in groups:
                    flattened.extend(group if isinstance(group, list) else [group])
        return flattened
    return []


def normalize_pageindex_results(raw_results: object, top_k: int) -> list[dict]:
    """Normalize common PageIndex response shapes without network access."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    output: list[dict[str, Any]] = []
    for row in _extract_rows(raw_results):
        if not isinstance(row, Mapping):
            continue
        content = next(
            (row.get(key) for key in ("content", "text", "relevant_content", "page_content", "document") if row.get(key)),
            None,
        )
        if not isinstance(content, str) or not content.strip():
            continue
        raw_meta = row.get("metadata")
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        metadata.setdefault("source", row.get("source") or row.get("url"))
        metadata.setdefault("title", row.get("title") or row.get("section_title"))
        metadata.setdefault("page", row.get("page") or row.get("page_number"))
        raw_score = row.get("score", row.get("relevance_score"))
        try:
            score = float(raw_score)
            trustworthy = math.isfinite(score)
        except (TypeError, ValueError):
            score, trustworthy = 0.0, False
        rank = len(output) + 1
        result = dict(row)
        result.update(
            {
                "content": content.strip(),
                "score": score if trustworthy else 0.0,
                "metadata": metadata,
                "retrieval_source": "pageindex",
                "source": "pageindex",
            }
        )
        if trustworthy:
            result["score_type"] = "provider"
        else:
            result["rank_score"] = 1.0 / rank
            result["score_type"] = "rank"
        output.append(result)
        if len(output) >= top_k:
            break
    return output


def _get_client() -> object:
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        raise PageIndexUnavailableError("Set PAGEINDEX_API_KEY to enable PageIndex retrieval")
    try:
        from pageindex import PageIndexClient  # type: ignore
    except (ImportError, AttributeError):
        try:
            from pageindex.client import PageIndexClient  # type: ignore
        except (ImportError, AttributeError) as exc:
            raise PageIndexUnavailableError("Install the 'pageindex' package to enable PageIndex retrieval") from exc
    try:
        return PageIndexClient(api_key=api_key)
    except TypeError as exc:
        raise PageIndexUnavailableError(
            "Installed PageIndexClient is incompatible; configure PAGEINDEX_BASE_URL for the REST adapter"
        ) from exc


def _request_pageindex_api(query: str, top_k: int, timeout: float) -> object:
    """Call the configured REST endpoint with an explicit timeout."""
    import requests

    base_url = os.getenv("PAGEINDEX_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not base_url:
        raise PageIndexUnavailableError("Set PAGEINDEX_BASE_URL to use the PageIndex REST adapter")
    response = requests.post(
        f"{base_url}/retrieval",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"index_id": os.environ["PAGEINDEX_INDEX_ID"].strip(), "query": query, "top_k": top_k},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _query_pageindex(query: str, top_k: int) -> object:
    """SDK compatibility boundary; monkeypatch this function in offline tests."""
    if not os.getenv("PAGEINDEX_API_KEY", "").strip():
        raise PageIndexUnavailableError("Set PAGEINDEX_API_KEY to enable PageIndex retrieval")
    index_id = os.getenv("PAGEINDEX_INDEX_ID", "").strip()
    if not index_id:
        raise PageIndexUnavailableError("Set PAGEINDEX_INDEX_ID to select a PageIndex index")
    try:
        timeout = float(os.getenv("PAGEINDEX_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise PageIndexUnavailableError("PAGEINDEX_TIMEOUT_SECONDS must be numeric") from exc
    if os.getenv("PAGEINDEX_BASE_URL", "").strip():
        return _request_pageindex_api(query, top_k, timeout)
    client = _get_client()
    for method_name in ("search", "query", "retrieve"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                return method(index_id=index_id, query=query, top_k=top_k, timeout=timeout)
            except TypeError:
                try:
                    return method(index_id, query, timeout=timeout)
                except TypeError as exc:
                    raise PageIndexUnavailableError(
                        "Installed PageIndex SDK search method does not accept a timeout"
                    ) from exc
    raise PageIndexUnavailableError("Installed PageIndex SDK has no supported search/query/retrieve method")


# Integration contract: Task 9 catches PageIndexError and preserves hybrid results.
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval using PageIndex as a hybrid-search fallback."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return normalize_pageindex_results(_query_pageindex(query.strip(), top_k), top_k)


def upload_documents(paths: list[str]) -> list[dict]:
    """Upload explicit paths using a compatible SDK method; never runs implicitly."""
    if not isinstance(paths, list):
        raise TypeError("paths must be a list")
    client = _get_client()
    upload = (
        getattr(client, "upload_document", None)
        or getattr(client, "submit_document", None)
        or getattr(client, "index", None)
    )
    if not callable(upload):
        raise PageIndexUnavailableError("Installed PageIndex SDK has no supported upload method")
    responses: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        response = upload(str(path))
        responses.append(dict(response) if isinstance(response, Mapping) else {"response": response})
    return responses
