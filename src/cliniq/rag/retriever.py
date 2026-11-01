"""Lexical retrieval over the ClinIQ evidence corpus.

Two methods share one interface: BM25 (rank_bm25, the default) and TF-IDF
cosine similarity (scikit-learn, kept as a baseline). Both are benchmarked in
evals/run_retrieval_eval.py.

``DenseRetriever`` is an experimental MedCPT dual-encoder implementation of
the same ``.search()`` interface. It is not yet benchmarked against the golden
set and needs the optional ``transformers`` and ``torch`` packages.

Usage (sanity-check a few queries):
    python -m cliniq.rag.retriever
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cliniq.config import CORPUS_DIR

CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Chunks are single sentences (~20-40 tokens), so without stopword filtering
# a couple of incidental "is"/"to" hits can outrank a chunk that contains the
# query's actual content word once.
_STOPWORDS = frozenset(
    """
a an the is are was were be been being to of in on for with as by at from
and or but if then than so that this these those it its it's they their
there here what which who whom how why when where not no nor do does did
can could will would shall should may might must about into over under
again further once more most other some such only own same
""".split()  # noqa: SIM905 (readable as a block)
)


def _destem(token: str) -> str:
    """Strip a trailing plural 's' (readmissions -> readmission).

    Deliberately crude: it does not handle '-ies' plurals, and on the current
    golden set it costs a little MRR (see ARCHITECTURE.md).
    """
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_destem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    with open(chunks_path) as f:
        return [json.loads(line) for line in f]


def _rank(chunks: list[dict], scores: np.ndarray, k: int) -> list[dict]:
    order = scores.argsort()[::-1][:k]
    return [{**chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(order)]


class Retriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks = load_chunks(chunks_path)
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self._tfidf_matrix = self._tfidf.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 5, method: str = "bm25") -> list[dict]:
        if method == "bm25":
            scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        elif method == "tfidf":
            scores = cosine_similarity(self._tfidf.transform([query]), self._tfidf_matrix)[0]
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")
        return _rank(self.chunks, scores, k)


class DenseRetriever:
    """Experimental MedCPT dual-encoder retrieval (ncbi/MedCPT-Query-Encoder
    and ncbi/MedCPT-Article-Encoder). Same ``.search()`` signature as
    ``Retriever``. Requires ``pip install transformers torch``."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        from transformers import AutoModel, AutoTokenizer

        self.chunks = load_chunks(chunks_path)
        self._q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self._q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self._d_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self._d_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self._doc_embeddings = self._embed_docs([c["text"] for c in self.chunks])

    def _embed_docs(self, texts: list[str]) -> Any:
        import torch

        with torch.no_grad():
            inputs = self._d_tok(texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
            return self._d_model(**inputs).last_hidden_state[:, 0, :]

    def search(self, query: str, k: int = 5, method: str = "dense") -> list[dict]:
        import torch

        with torch.no_grad():
            q_inputs = self._q_tok([query], truncation=True, padding=True, return_tensors="pt", max_length=64)
            q_emb = self._q_model(**q_inputs).last_hidden_state[:, 0, :]
        scores = torch.nn.functional.cosine_similarity(q_emb, self._doc_embeddings).numpy()
        return _rank(self.chunks, scores, k)


if __name__ == "__main__":
    retriever = Retriever()
    for q in (
        "does missing a follow-up appointment increase readmission risk",
        "is insulin use linked to hospital readmission",
        "why do hospitals get penalized for readmissions",
    ):
        print(f"\nQuery: {q}")
        for hit in retriever.search(q, k=3):
            print(f"  [{hit['score']:.2f}] {hit['title']} -- {hit['text'][:90]}...")
