"""Task 10: grounded LLM generation with stable citations.

``generate_with_citation(query, context_chunks)`` consumes normalized retrieval
chunks and returns a string. Providers (``openai`` or offline ``mock``) are
selected with ``LLM_PROVIDER`` and initialized lazily. Configuration includes
``LLM_MODEL``, ``LLM_API_KEY``, ``LLM_TEMPERATURE``, ``LLM_TOP_P``, and
``LLM_TIMEOUT_SECONDS``. Empty/invalid generations fail closed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

UNVERIFIABLE_RESPONSE = "I cannot verify this information"

SYSTEM_PROMPT = """You are a grounded university-services question-answering assistant.
Use only facts explicitly stated in the supplied context.
Every factual claim must be immediately followed by one of the exact citation labels supplied in the context.
Never invent, alter, or infer a source name or year.
If the context is insufficient, reply exactly: I cannot verify this information
"""

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_CITATION_RE = re.compile(r"\[[^\[\]\n]+,\s*(?:(?:19|20)\d{2}|n\.d\.)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Reorder ranked chunks so high-value chunks occupy both context edges."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    copied = [dict(chunk) if isinstance(chunk, Mapping) else chunk for chunk in chunks]
    if len(copied) <= 2:
        return copied
    return copied[::2] + copied[1::2][::-1]


def _metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    raw = chunk.get("metadata")
    return dict(raw) if isinstance(raw, Mapping) else {}


def build_citation_label(chunk: Mapping[str, Any]) -> str:
    """Build ``[Source, Year]`` strictly from available chunk metadata."""
    metadata = _metadata(chunk)
    source: Any = None
    for key in ("citation", "title", "source", "url", "filename", "path"):
        if metadata.get(key):
            source = metadata[key]
            break
    if not source:
        source = (
            chunk.get("citation")
            or chunk.get("title")
            or chunk.get("source")
            or chunk.get("url")
            or chunk.get("filename")
            or chunk.get("path")
        )
    source_text = str(source).strip() if source else "Unknown Source"
    if source_text != "Unknown Source" and ("/" in source_text or "\\" in source_text):
        source_text = Path(source_text).name or source_text

    year: str | None = None
    explicit_year = metadata.get("year")
    if explicit_year not in (None, ""):
        match = _YEAR_RE.search(str(explicit_year))
        year = match.group(1) if match else None
    if year is None:
        for key in ("published_date", "date", "crawl_date"):
            match = _YEAR_RE.search(str(metadata.get(key, "")))
            if match:
                year = match.group(1)
                break
    return f"[{source_text}, {year or 'n.d.'}]"


def format_context(chunks: list[dict]) -> str:
    """Format reordered chunks with explicit source metadata and citation labels."""
    sections: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, Mapping):
            continue
        content = chunk.get("content") or chunk.get("text") or chunk.get("page_content")
        if not isinstance(content, str) or not content.strip():
            continue
        metadata = _metadata(chunk)
        sections.append(
            "\n".join(
                (
                    f"[CONTEXT {index}]",
                    f"Citation: {build_citation_label(chunk)}",
                    f"Source: {metadata.get('source') or metadata.get('url') or ''}",
                    f"Title: {metadata.get('title') or ''}",
                    f"URL: {metadata.get('url') or ''}",
                    "Content:",
                    content.strip(),
                )
            )
        )
    return "\n\n".join(sections)


def citations_are_valid(answer: str, allowed_labels: set[str]) -> bool:
    """Return false when the model emits a citation absent from supplied context."""
    return all(citation in allowed_labels for citation in _CITATION_RE.findall(answer))


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Lazy provider adapter kept monkeypatchable for offline unit tests."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().casefold()
    if provider == "mock":
        return UNVERIFIABLE_RESPONSE
    if provider != "openai":
        raise ValueError("LLM_PROVIDER must be openai or mock")
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY to use the OpenAI provider")
    model = os.getenv("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError("Set LLM_MODEL to use the OpenAI provider")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Low temperature reduces hallucination; top_p keeps wording natural while bounded.
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        top_p=float(os.getenv("LLM_TOP_P", "0.9")),
    )
    # Generation LLM_TOP_K is intentionally omitted: OpenAI's API does not support it.
    return response.choices[0].message.content or ""


# Integration contract: Task 9 passes normalized chunks; this boundary never retrieves.
def generate_with_citation(query: str, context_chunks: list[dict]) -> str:
    """Generate a grounded answer whose citations come only from supplied chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(context_chunks, list):
        raise TypeError("context_chunks must be a list")
    if not context_chunks:
        return UNVERIFIABLE_RESPONSE
    reordered = reorder_for_llm(context_chunks)
    context = format_context(reordered)
    if not context:
        return UNVERIFIABLE_RESPONSE
    labels = {build_citation_label(chunk) for chunk in reordered if isinstance(chunk, Mapping)}
    user_prompt = f"Context:\n{context}\n\nQuestion: {query.strip()}"
    answer = (_call_llm(SYSTEM_PROMPT, user_prompt) or "").strip()
    if not answer:
        return UNVERIFIABLE_RESPONSE
    if citations_are_valid(answer, labels):
        return answer

    retry_prompt = (
        user_prompt
        + "\n\nYour prior answer used an unavailable citation. Rewrite using only these exact labels: "
        + "; ".join(sorted(labels))
    )
    retried = (_call_llm(SYSTEM_PROMPT, retry_prompt) or "").strip()
    return retried if retried and citations_are_valid(retried, labels) else UNVERIFIABLE_RESPONSE
