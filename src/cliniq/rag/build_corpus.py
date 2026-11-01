"""Flatten the curated source list (data/corpus/sources.json) into
data/corpus/chunks.jsonl, the unit the retriever indexes.

Each source is a named, dated clinical-literature or policy document distilled
into short single-claim passages. Every chunk carries its source's citation
metadata so the agent can attribute each sentence it writes.

Usage:
    python -m cliniq.rag.build_corpus
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cliniq.config import CORPUS_DIR

log = logging.getLogger(__name__)

CITATION_FIELDS = ("source_id", "title", "authors", "year", "url", "source_type")


def build_chunks(sources: list[dict]) -> list[dict]:
    return [
        {"chunk_id": f"{source['source_id']}_{i}", **{k: source[k] for k in CITATION_FIELDS}, "text": passage}
        for source in sources
        for i, passage in enumerate(source["passages"])
    ]


def run(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    with open(corpus_dir / "sources.json") as f:
        sources = json.load(f)
    chunks = build_chunks(sources)
    out_path = corpus_dir / "chunks.jsonl"
    with open(out_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    log.info("Wrote %d chunks from %d sources to %s", len(chunks), len(sources), out_path)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run()
