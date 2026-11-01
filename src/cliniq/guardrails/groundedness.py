"""Groundedness guardrail for agent narratives.

A cheap, deterministic check that runs on every agent call, deliberately not
a second LLM call (which can fail the same way the first one did):

1. Sentence-level claim overlap: each non-trivial sentence must share at
   least OVERLAP_THRESHOLD of its tokens with some retrieved chunk.
2. Drug-name leakage: a diabetes medication named in the narrative but absent
   from the retrieved context is flagged. Drugs that are one of the patient's
   own SHAP drivers are reported separately as a corpus-coverage gap, not a
   fabrication.

Known limitations (see ARCHITECTURE.md): lexical overlap cannot detect
negated or direction-flipped claims, sentences under SENTENCE_MIN_TOKENS
content tokens are not scored, and the drug lexicon covers only the
medications present in the training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cliniq.rag.retriever import tokenize

# Medication columns that survive the variance filter in data_prep.py.
KNOWN_DRUGS = {
    "metformin",
    "repaglinide",
    "nateglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
}

SENTENCE_MIN_TOKENS = 4
OVERLAP_THRESHOLD = 0.4
GROUNDED_RATIO_THRESHOLD = 0.8

# Lines that restate an already-computed value (a driver's feature=value, the
# fixed disclaimer) rather than make a clinical claim. Scoring them against
# the corpus would only measure that a label shares no vocabulary with any
# passage. These patterns match TemplateProvider's structured output.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^Contributing factor --.*:$"),
    re.compile(r"^Estimated 30-day readmission risk:.*$"),
    re.compile(r"^This summary is generated from retrieved sources only.*$"),
    re.compile(r"^No directly relevant passage was retrieved.*$"),
]
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class GroundednessReport:
    grounded: bool
    grounded_ratio: float
    sentence_results: list[dict] = field(default_factory=list)
    fabricated_drug_mentions: list[str] = field(default_factory=list)
    unsupported_known_drivers: list[str] = field(default_factory=list)


def _is_boilerplate(sentence: str) -> bool:
    return any(p.match(sentence) for p in _BOILERPLATE_PATTERNS)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _overlap_ratio(sentence_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def check(narrative: str, retrieved_chunks: list[dict], known_entities: set[str] | None = None) -> GroundednessReport:
    chunk_token_sets = [set(tokenize(c["text"])) for c in retrieved_chunks]
    all_retrieved_tokens = set().union(*chunk_token_sets)
    known_entities = known_entities or set()

    sentence_results = []
    for sentence in _split_sentences(narrative):
        if _is_boilerplate(sentence):
            continue
        s_tokens = set(tokenize(sentence))
        if len(s_tokens) < SENTENCE_MIN_TOKENS:
            continue
        best = max((_overlap_ratio(s_tokens, ct) for ct in chunk_token_sets), default=0.0)
        sentence_results.append(
            {"sentence": sentence, "max_overlap": round(best, 2), "grounded": best >= OVERLAP_THRESHOLD}
        )

    checked = len(sentence_results)
    grounded_count = sum(1 for r in sentence_results if r["grounded"])
    grounded_ratio = grounded_count / checked if checked else 1.0

    mentioned = set(tokenize(narrative)) & KNOWN_DRUGS
    unsupported = mentioned - all_retrieved_tokens
    fabricated = sorted(unsupported - known_entities)
    unsupported_known = sorted(unsupported & known_entities)

    return GroundednessReport(
        grounded=grounded_ratio >= GROUNDED_RATIO_THRESHOLD and not fabricated,
        grounded_ratio=round(grounded_ratio, 2),
        sentence_results=sentence_results,
        fabricated_drug_mentions=fabricated,
        unsupported_known_drivers=unsupported_known,
    )
