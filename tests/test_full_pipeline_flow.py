"""Deterministic Markdown-to-citation flow without external services."""

from __future__ import annotations

import math
import shutil

import pytest

from src import task10_generation as generation
from src.task4_chunking_indexing import chunk_documents, embed_chunks, load_documents
from src.task5_semantic_search import semantic_search_with_dependencies
from src.task6_lexical_search import lexical_search_with_dependencies
from src.task7_reranking import rerank
from src.task9_retrieval_pipeline import retrieve_with_dependencies

pytestmark = pytest.mark.e2e


class _Vector(list):
    def tolist(self):
        return list(self)


class _Embedding:
    def encode(self, values, show_progress_bar=False):
        def vector(text):
            lowered = text.casefold()
            return _Vector([float("thử việc" in lowered), float("nghỉ" in lowered), 0.1])

        return [vector(value) for value in values] if isinstance(values, list) else vector(values)


class _MemoryCollection:
    def __init__(self, chunks):
        self.chunks = chunks

    def query(self, query_embeddings, n_results, include):
        query = query_embeddings[0]

        def cosine(item):
            vector = item["embedding"]
            denominator = math.sqrt(sum(x*x for x in query)) * math.sqrt(sum(x*x for x in vector))
            return sum(a*b for a, b in zip(query, vector)) / denominator if denominator else 0.0

        ordered = sorted(self.chunks, key=cosine, reverse=True)[:n_results]
        return {
            "documents": [[item["content"] for item in ordered]],
            "metadatas": [[item["metadata"] for item in ordered]],
            "distances": [[1.0 - cosine(item) for item in ordered]],
        }


class _BM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        terms = set(query_tokens)
        return [float(sum(token in item["content"].casefold().split() for token in terms)) for item in self.corpus]


def test_markdown_to_retrieval_to_grounded_citation(tmp_path, monkeypatch):
    standardized = tmp_path / "standardized" / "legal"
    standardized.mkdir(parents=True)
    shutil.copyfile("tests/fixtures/labour_law/sample_labour_law.md", standardized / "sample.md")
    documents = load_documents(tmp_path / "standardized")
    chunks = chunk_documents(documents)
    embedded = embed_chunks(chunks, model=_Embedding())
    collection = _MemoryCollection(embedded)
    model = _Embedding()

    semantic = lambda query, top_k: semantic_search_with_dependencies(
        query, model=model, collection=collection, top_k=top_k
    )
    lexical = lambda query, top_k: lexical_search_with_dependencies(
        query, chunks, _BM25(chunks), top_k=top_k
    )
    monkeypatch.setenv("RERANK_BACKEND", "rrf")
    results = retrieve_with_dependencies(
        "Thử việc là gì?",
        semantic_search_fn=semantic,
        lexical_search_fn=lexical,
        rerank_fn=rerank,
        pageindex_search_fn=lambda query, top_k: [],
        top_k=3,
    )
    monkeypatch.setattr(
        generation,
        "_call_llm",
        lambda system_prompt, user_prompt: (
            "Các bên có thể thỏa thuận về thử việc "
            "[Tài liệu thử nghiệm luật lao động, n.d.]."
        ),
    )
    answer = generation.generate_with_citation("Thử việc là gì?", results)
    assert results and {"content", "metadata"} <= results[0].keys()
    assert "[Tài liệu thử nghiệm luật lao động, n.d.]" in answer

