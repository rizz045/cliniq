from __future__ import annotations

import json

import pytest

from cliniq.config import CORPUS_DIR
from cliniq.rag.build_corpus import build_chunks
from cliniq.rag.retriever import Retriever, tokenize
from evals.run_retrieval_eval import evaluate, load_golden_set

# CI gate: retrieval changes must not drop quality below these floors.
MIN_BM25_MRR = 0.75
MIN_BM25_RECALL_AT_5 = 0.75


@pytest.fixture(scope="module")
def corpus_retriever(tmp_path_factory) -> Retriever:
    sources = json.loads((CORPUS_DIR / "sources.json").read_text())
    path = tmp_path_factory.mktemp("corpus") / "chunks.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in build_chunks(sources)))
    return Retriever(path)


def test_tokenize_drops_stopwords_and_plural_s():
    assert tokenize("The readmissions of patients") == ["readmission", "patient"]


def test_tokenize_keeps_double_s_words():
    assert tokenize("class process") == ["class", "process"]


def test_every_chunk_has_citation_metadata(corpus_retriever):
    for chunk in corpus_retriever.chunks:
        assert chunk["authors"] and chunk["year"] and chunk["url"].startswith("https://")


def test_golden_set_references_existing_chunks(corpus_retriever):
    ids = {c["chunk_id"] for c in corpus_retriever.chunks}
    for item in load_golden_set():
        assert set(item["relevant_chunk_ids"]) <= ids, item["query"]


def test_bm25_ranks_obvious_match_first(corpus_retriever):
    [top] = corpus_retriever.search("left against medical advice", k=1)
    assert "against medical advice" in top["text"].lower()


def test_unknown_method_raises(corpus_retriever):
    with pytest.raises(ValueError):
        corpus_retriever.search("insulin", method="nope")


def test_bm25_retrieval_quality_floor(corpus_retriever):
    result = evaluate(corpus_retriever, load_golden_set(), "bm25")
    assert result["mrr"] >= MIN_BM25_MRR
    assert result["recall@5"] >= MIN_BM25_RECALL_AT_5
