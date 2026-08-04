"""Focused offline tests for paths, adapters, retrieval, and citations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src import task10_generation as generation
from src.integration_adapters import make_candidate_id, normalize_candidate
from src.task3_convert_markdown import convert_legal_docs, convert_news_articles
from src.task4_chunking_indexing import chunk_documents, load_documents
from src.task5_semantic_search import semantic_search_with_dependencies
from src.task6_lexical_search import lexical_search_with_dependencies
from src.task7_reranking import rerank
from src.task9_retrieval_pipeline import RetrievalPipelineError, retrieve_with_dependencies

pytestmark = pytest.mark.integration


class _Conversion:
    text_content = "Nội dung pháp luật lao động tiếng Việt"


class _Converter:
    def convert(self, path):
        return _Conversion()


def test_task1_2_paths_are_discovered_by_task3_and_task4(tmp_path):
    landing = tmp_path / "data" / "landing"
    legal = landing / "legal"
    news = landing / "news"
    standardized = tmp_path / "data" / "standardized"
    legal.mkdir(parents=True)
    news.mkdir(parents=True)
    (legal / "bo-luat.pdf").write_bytes(b"fake fixture")
    (news / "thu-viec.json").write_text(
        json.dumps(
            {
                "title": "Tài liệu thử việc",
                "url": "https://example.test",
                "content_markdown": "Quy định thử việc.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    convert_legal_docs(legal, standardized / "legal", converter=_Converter())
    convert_news_articles(news, standardized / "news")
    documents = load_documents(standardized)
    assert len(documents) == 2
    assert {item["metadata"]["type"] for item in documents} == {"legal", "news"}
    assert all(item["content"].strip() for item in documents)


def test_candidate_adapter_preserves_alias_metadata_and_input():
    original = {"page_content": "Nội dung thử việc", "distance": 0.1, "metadata": {"title": "Nguồn"}}
    before = copy.deepcopy(original)
    adapted = normalize_candidate(original)
    assert original == before
    assert adapted["content"] == "Nội dung thử việc"
    assert adapted["metadata"]["title"] == "Nguồn"
    assert adapted["raw_distance"] == 0.1
    assert adapted["score_type"] == "distance"
    assert make_candidate_id(adapted) == make_candidate_id(copy.deepcopy(adapted))


class _Vector(list):
    def tolist(self):
        return list(self)


class _Model:
    def encode(self, query):
        return _Vector([1.0, 0.0])


class _Collection:
    def query(self, **kwargs):
        return {
            "documents": [["Thời gian thử việc được các bên thỏa thuận."]],
            "metadatas": [[{"chunk_id": "chunk-1", "title": "Tài liệu thử việc"}]],
            "distances": [[0.1]],
        }


class _BM25:
    def get_scores(self, query):
        return [12.5, 8.0]


def test_task4_5_6_schema_wrappers_keep_calibrated_scores():
    semantic = semantic_search_with_dependencies(
        "thử việc", model=_Model(), collection=_Collection(), top_k=5
    )
    corpus = [
        {"content": "Thời gian thử việc được các bên thỏa thuận.", "metadata": {"chunk_id": "chunk-1"}},
        {"content": "Người lao động có quyền nghỉ hằng năm.", "metadata": {"chunk_id": "chunk-2"}},
    ]
    lexical = lexical_search_with_dependencies("thử việc", corpus, _BM25(), top_k=5)
    assert semantic[0]["semantic_score"] == pytest.approx(0.9)
    assert semantic[0]["raw_distance"] == 0.1
    assert lexical[0]["lexical_score"] == 12.5
    assert lexical[0]["metadata"]["chunk_id"] == "chunk-1"


def _offline_rerank(query, candidates, top_k=5):
    return [dict(item, rerank_score=1.0 / (index + 1)) for index, item in enumerate(candidates[:top_k])]


def test_task5_6_7_9_merge_threshold_and_no_mutation(monkeypatch):
    semantic_input = [{
        "content": "Thời gian thử việc được các bên thỏa thuận.",
        "score": 0.9,
        "semantic_score": 0.9,
        "metadata": {"chunk_id": "chunk-1", "title": "Tài liệu thử việc"},
    }]
    lexical_input = [
        {
            "content": semantic_input[0]["content"],
            "score": 12.5,
            "lexical_score": 12.5,
            "metadata": copy.deepcopy(semantic_input[0]["metadata"]),
        },
        {
            "content": "Người lao động có quyền nghỉ hằng năm.",
            "score": 8.0,
            "lexical_score": 8.0,
            "metadata": {"chunk_id": "chunk-2", "title": "Tài liệu nghỉ hằng năm"},
        },
    ]
    before = copy.deepcopy((semantic_input, lexical_input))
    page_calls = []
    results = retrieve_with_dependencies(
        "thử việc",
        semantic_search_fn=lambda query, top_k: semantic_input,
        lexical_search_fn=lambda query, top_k: lexical_input,
        rerank_fn=_offline_rerank,
        pageindex_search_fn=lambda query, top_k: page_calls.append(True) or [],
        top_k=2,
        score_threshold=0.3,
    )
    assert (semantic_input, lexical_input) == before
    assert len(results) == 2
    assert results[0]["fusion_score"] == pytest.approx(2 / 61)
    assert results[0]["semantic_score"] == 0.9
    assert results[0]["lexical_score"] == 12.5
    assert page_calls == []  # RRF ≈ .03 is not compared with the .3 semantic threshold.


def test_task9_fallback_and_partial_failures():
    page_calls = []
    weak = retrieve_with_dependencies(
        "query",
        semantic_search_fn=lambda query, top_k: [{"content": "weak", "score": 0.1, "semantic_score": 0.1}],
        lexical_search_fn=lambda query, top_k: [],
        rerank_fn=_offline_rerank,
        pageindex_search_fn=lambda query, top_k: page_calls.append(True) or [{"content": "fallback", "score": 0.0}],
    )
    assert page_calls and weak

    def broken(query, top_k):
        raise RuntimeError("secret")

    lexical_only = retrieve_with_dependencies(
        "query",
        semantic_search_fn=broken,
        lexical_search_fn=lambda query, top_k: [{"content": "lexical", "score": 2.0}],
        rerank_fn=_offline_rerank,
        pageindex_search_fn=broken,
    )
    assert lexical_only
    with pytest.raises(RetrievalPipelineError) as error:
        retrieve_with_dependencies(
            "query",
            semantic_search_fn=broken,
            lexical_search_fn=broken,
            rerank_fn=_offline_rerank,
            pageindex_search_fn=broken,
        )
    assert "secret" not in str(error.value)


def test_task9_output_flows_directly_to_task10(monkeypatch):
    chunks = [{
        "content": "Các bên có thể thỏa thuận về thử việc.",
        "score": 0.9,
        "metadata": {"title": "Tài liệu thử nghiệm luật lao động", "source": "Nguồn thử nghiệm"},
    }]
    prompts = []

    def fake_llm(system_prompt, user_prompt):
        prompts.append(user_prompt)
        return "Các bên có thể thỏa thuận về thử việc [Tài liệu thử nghiệm luật lao động, n.d.]."

    monkeypatch.setattr(generation, "_call_llm", fake_llm)
    answer = generation.generate_with_citation("Thử việc là gì?", chunks)
    assert "[Tài liệu thử nghiệm luật lao động, n.d.]" in answer
    assert chunks[0]["content"] in prompts[0]
    assert "Nguồn thử nghiệm" in prompts[0]
    assert len(generation.reorder_for_llm(chunks)) == len(chunks)

