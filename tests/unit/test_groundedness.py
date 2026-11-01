from __future__ import annotations

import pytest

from cliniq.guardrails import groundedness

CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "Longer length of stay was associated with higher readmission odds.",
        "authors": "Karunakaran A",
        "year": 2018,
    }
]


def test_unsupported_claim_is_flagged():
    report = groundedness.check("Risk is driven by a rare genetic condition and an unusual blood type.", CHUNKS)
    assert not report.grounded
    assert report.grounded_ratio < 0.8


def test_supported_claim_passes():
    report = groundedness.check(
        "Longer length of stay was associated with higher readmission odds (Karunakaran A, 2018).", CHUNKS
    )
    assert report.grounded


def test_drug_absent_from_context_and_drivers_is_fabrication():
    report = groundedness.check(
        "Longer length of stay was associated with higher readmission odds. Consider rosiglitazone.",
        CHUNKS,
        known_entities=set(),
    )
    assert report.fabricated_drug_mentions == ["rosiglitazone"]
    assert not report.grounded


def test_drug_that_is_a_patient_driver_is_a_coverage_gap_not_fabrication():
    report = groundedness.check(
        "Longer length of stay was associated with higher readmission odds. Contributing factor -- insulin = Up.",
        CHUNKS,
        known_entities={"insulin"},
    )
    assert report.fabricated_drug_mentions == []
    assert report.unsupported_known_drivers == ["insulin"]


def test_boilerplate_lines_are_not_scored_as_claims():
    report = groundedness.check(
        "Contributing factor -- age_midpoint = 75:\n"
        "Longer length of stay was associated with higher readmission odds (Karunakaran A, 2018).",
        CHUNKS,
    )
    assert report.grounded_ratio == 1.0


# Known limitations of lexical overlap, documented as strict xfails so that a
# stronger checker (entailment / structured citations) flips them to passing.


@pytest.mark.xfail(strict=True, reason="lexical overlap cannot detect negation")
def test_negated_claim_is_flagged():
    report = groundedness.check("Longer length of stay was not associated with higher readmission odds.", CHUNKS)
    assert not report.grounded


@pytest.mark.xfail(strict=True, reason="drug lexicon only covers medications in the training data")
def test_drug_outside_training_lexicon_is_flagged_as_fabricated():
    report = groundedness.check(
        "Longer length of stay was associated with higher readmission odds. Start empagliflozin.", CHUNKS
    )
    assert report.fabricated_drug_mentions == ["empagliflozin"]


@pytest.mark.xfail(strict=True, reason="sentences under SENTENCE_MIN_TOKENS are not scored")
def test_short_treatment_directive_is_flagged():
    report = groundedness.check("Stop insulin now.", CHUNKS, known_entities={"insulin"})
    assert not report.grounded
