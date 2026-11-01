from __future__ import annotations

import pytest

from cliniq.llm.providers import AnthropicProvider, LLMProvider, TemplateProvider, get_provider
from evals.rubrics import CITATION_RE, DOSAGE_RE, grade_heuristic, grade_with_llm_judge

CONTEXT = {
    "risk_score": 0.25,
    "base_rate": 0.11,
    "drivers": [
        {"feature": "number_inpatient", "value": 3, "shap_contribution": 0.4},
        {"feature": "diag_1_group", "value": "Neoplasms", "shap_contribution": 0.1},
    ],
    "retrieved_by_driver": {
        "number_inpatient": [{"text": "Prior admissions predict readmission.", "authors": "Doe J", "year": 2020}]
    },
}


def test_get_provider_defaults_to_template(monkeypatch):
    monkeypatch.delenv("CLINIQ_LLM_PROVIDER", raising=False)
    assert isinstance(get_provider(), TemplateProvider)


def test_get_provider_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_provider("gpt-17")


def test_anthropic_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_template_narrative_cites_evidence_and_admits_gaps():
    narrative = TemplateProvider().generate("", CONTEXT)
    assert "25.0%" in narrative and "11.0%" in narrative
    assert '"Prior admissions predict readmission." (Doe J, 2020)' in narrative
    assert "No directly relevant passage was retrieved" in narrative
    assert "clinician" in narrative


def test_template_narrative_passes_heuristic_rubric():
    narrative = TemplateProvider().generate("", CONTEXT)
    assert grade_heuristic(narrative, CONTEXT)["score"] == 1.0


@pytest.mark.parametrize("text", ["take 10 mg daily", "give 5 units", "20mcg"])
def test_dosage_regex_detects_doses(text):
    assert DOSAGE_RE.search(text)


def test_citation_regex_matches_author_year():
    assert CITATION_RE.search("as reported (Karunakaran A, Zhao H, 2018).")
    assert not CITATION_RE.search("no citation here")


def test_llm_judge_handles_invalid_json():
    class _Garbage(LLMProvider):
        def generate(self, prompt, context=None):
            return "not json"

    result = grade_with_llm_judge("narrative", CONTEXT, _Garbage())
    assert result["score"] is None and "error" in result


def test_llm_judge_scores_valid_verdicts():
    class _Judge(LLMProvider):
        def generate(self, prompt, context=None):
            return '{"cites_sources": true, "no_drug_dosing": true}'

    assert grade_with_llm_judge("narrative", CONTEXT, _Judge())["score"] == 0.5
