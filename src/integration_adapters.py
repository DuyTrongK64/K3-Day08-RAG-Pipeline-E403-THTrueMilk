"""Non-mutating adapters between independently implemented task modules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def get_content(item: object) -> str:
    if isinstance(item, Mapping):
        for key in ("content", "text", "page_content", "document", "body"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("page_content", "content", "text"):
        value = getattr(item, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_metadata(item: object) -> dict[str, Any]:
    value = item.get("metadata") if isinstance(item, Mapping) else getattr(item, "metadata", None)
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_candidate(item: object) -> dict[str, Any]:
    """Copy common document/result shapes into the minimum retrieval schema."""
    result = dict(item) if isinstance(item, Mapping) else {}
    result["content"] = get_content(item)
    result["metadata"] = get_metadata(item)
    score = next(
        (result.get(key) for key in ("score", "similarity", "relevance_score") if result.get(key) is not None),
        0.0,
    )
    try:
        result["score"] = float(score)
    except (TypeError, ValueError):
        result["score"] = 0.0
    if result.get("distance") is not None:
        result.setdefault("raw_distance", result["distance"])
        result.setdefault("score_type", "distance")
    return result


def make_candidate_id(item: Mapping[str, Any]) -> str:
    metadata = get_metadata(item)
    for key in ("chunk_id", "id"):
        if metadata.get(key) not in (None, ""):
            return f"{key}:{metadata[key]}"
    if metadata.get("document_id") not in (None, ""):
        return f"document:{metadata['document_id']}:{metadata.get('page', '')}:{metadata.get('section', '')}"
    canonical = "\x1f".join(
        (
            str(metadata.get("source") or metadata.get("url") or metadata.get("file_path") or ""),
            str(metadata.get("page") or ""),
            str(metadata.get("section") or ""),
            " ".join(get_content(item).split()).casefold(),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_front_matter(content: str) -> dict[str, Any]:
    """Parse the simple JSON-valued YAML emitted by Task 3 without a YAML dependency."""
    if not content.startswith("---\n"):
        return {}
    closing = content.find("\n---", 4)
    if closing < 0:
        return {}
    metadata: dict[str, Any] = {}
    for line in content[4:closing].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            metadata[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            metadata[key.strip()] = value.strip().strip('"')
    return metadata

