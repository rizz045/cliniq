"""Retrieval quality against the hand-labeled golden set
(evals/golden_retrieval_set.jsonl), BM25 vs. TF-IDF.

Metrics at k in {3, 5}: recall@k, precision@k, and MRR (rank of the first
relevant chunk in the full ranking).

Usage:
    python -m evals.run_retrieval_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from cliniq.config import REPORTS_DIR
from cliniq.rag.retriever import Retriever

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_retrieval_set.jsonl"


def load_golden_set(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate(retriever: Retriever, golden: list[dict], method: str, ks: tuple[int, ...] = (3, 5)) -> dict:
    recall: dict[int, list[float]] = {k: [] for k in ks}
    precision: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks = []

    for item in golden:
        relevant = set(item["relevant_chunk_ids"])
        ranked_ids = [h["chunk_id"] for h in retriever.search(item["query"], k=len(retriever.chunks), method=method)]
        first_hit = next((i + 1 for i, cid in enumerate(ranked_ids) if cid in relevant), None)
        reciprocal_ranks.append(1 / first_hit if first_hit else 0.0)
        for k in ks:
            hits = len(set(ranked_ids[:k]) & relevant)
            recall[k].append(hits / len(relevant))
            precision[k].append(hits / k)

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3)

    return {
        "method": method,
        "mrr": mean(reciprocal_ranks),
        **{f"recall@{k}": mean(recall[k]) for k in ks},
        **{f"precision@{k}": mean(precision[k]) for k in ks},
    }


def run(retriever: Retriever | None = None) -> dict:
    retriever = retriever or Retriever()
    golden = load_golden_set()
    results: dict[str, dict] = {m: evaluate(retriever, golden, m) for m in ("bm25", "tfidf")}
    results["_meta"] = {"n_queries": len(golden), "n_chunks": len(retriever.chunks)}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "retrieval_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
