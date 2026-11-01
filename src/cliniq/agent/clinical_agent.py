"""Orchestration: risk model -> SHAP drivers -> retrieval -> generation -> guardrail.

A plain, explicit call sequence rather than an agent framework: the steps are
fixed, so ``assess()`` can be read top to bottom with no hidden control flow.
A framework would earn its place if the LLM needed to choose the next tool
(e.g. decide whether more retrieval is needed).

Per call:
  1. predict   LightGBM + isotonic calibration
  2. explain   top-3 SHAP drivers for this patient
  3. retrieve  one query per driver, filtered by a per-feature relevance gate
  4. generate  LLMProvider.generate()
  5. check     groundedness + drug-name check; on failure, retry once with a
               stricter prompt, then return grounded=False so the caller can
               route the narrative to human review.

Usage (two sample test-set patients):
    python -m cliniq.agent.clinical_agent
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap

from cliniq.config import MODELS_DIR, PROCESSED_DIR
from cliniq.guardrails import groundedness
from cliniq.llm.providers import LLMProvider, get_provider
from cliniq.rag.retriever import Retriever
from cliniq.risk_model import top_risk_drivers

QUERY_TEMPLATES: dict[str, str] = {
    "number_inpatient": "prior inpatient hospital admissions and risk of readmission in diabetes",
    "number_emergency": "emergency department visit history and risk of readmission in diabetes",
    "number_outpatient": "outpatient follow-up visits and risk of hospital readmission in diabetes",
    "discharge_disposition_id_label": "effect of discharge disposition {value} on readmission risk in diabetes",
    "diag_1_group": "{value} diagnosis and readmission risk in diabetes patients",
    "diag_2_group": "{value} diagnosis and readmission risk in diabetes patients",
    "diag_3_group": "{value} diagnosis and readmission risk in diabetes patients",
    "number_diagnoses": "number of diagnoses and comorbidity burden as a readmission risk factor in diabetes",
    "age_midpoint": "patient age and risk of hospital readmission in diabetes",
    "time_in_hospital": "length of hospital stay and risk of readmission in diabetes",
    "insulin": "insulin therapy and risk of hospital readmission in diabetes",
    "A1Cresult": "HbA1c measurement and risk of hospital readmission in diabetes",
    "diabetesMed": "diabetes medication management and risk of hospital readmission",
    "num_medications": "number of medications and risk of hospital readmission in diabetes",
}

# Relevance gate. On short passages, generic terms ("readmission", "diabetes")
# dominate lexical scores, so the top hit can be on-topic but wrong for the
# specific driver: an SNF discharge was once paired with evidence about
# leaving against medical advice. The groundedness check cannot catch this,
# since the narrative faithfully quotes the (wrong) chunk. A candidate chunk
# therefore counts as evidence only if it contains a feature-specific
# keyword. An empty list means "no corpus coverage": the agent says so rather
# than citing adjacent evidence. Features absent from the table are treated
# the same way (fail closed).
RELEVANCE_KEYWORDS: dict[str, list[str]] = {
    "number_inpatient": ["admission", "discharge", "hospitalization", "utilization", "prior"],
    "number_emergency": ["emergency", "department"],
    "number_outpatient": ["outpatient", "primary care", "follow-up", "followup"],
    "number_diagnoses": ["comorbid", "diagnos"],
    "time_in_hospital": ["length of stay", "length-of-stay"],
    "insulin": ["insulin"],
    "A1Cresult": ["hba1c", "a1c", "glyc"],
    "diabetesMed": ["medication", "glyc", "intensif"],
    "num_medications": ["medication", "drug"],
    "age_midpoint": ["age"],
}

# Value-dependent gates. The corpus covers leaving against medical advice but
# no other discharge disposition, and discusses two diagnosis groups by
# clinical name ("heart failure", "renal disease") rather than bucket label.
DISCHARGE_DISPOSITION_KEYWORDS: dict[str, list[str]] = {
    "Left AMA": ["advice", "ama"],
}
DIAG_GROUP_KEYWORDS: dict[str, list[str]] = {
    "Circulatory": ["heart failure", "cardiovascular"],
    "Genitourinary": ["renal disease", "renal"],
}

_PATIENT_ID_RE = re.compile(r"[^A-Za-z0-9._-]")
_PATIENT_ID_MAX_LEN = 64


def _default_query(feature: str, value: Any) -> str:
    template = QUERY_TEMPLATES.get(feature, "clinical significance of {feature} ({value}) in diabetes readmission risk")
    return template.format(feature=feature, value=value)


def _relevance_keywords(feature: str, value: Any) -> list[str]:
    """Keywords a chunk must contain to count as evidence for this driver."""
    if feature == "discharge_disposition_id_label":
        return DISCHARGE_DISPOSITION_KEYWORDS.get(str(value), [])
    if feature in ("diag_1_group", "diag_2_group", "diag_3_group"):
        return DIAG_GROUP_KEYWORDS.get(str(value), [])
    return RELEVANCE_KEYWORDS.get(feature, [])


def _mentions(text: str, keywords: list[str]) -> bool:
    """Word-start keyword match: "age" matches "age"/"aged" but not
    "percentage" or "management"; prefixes like "comorbid" still match."""
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}", lowered) for kw in keywords)


def retrieve_evidence(retriever: Retriever, feature: str, value: Any, k: int = 3) -> list[dict]:
    """Up to k corpus chunks that pass the relevance gate for one driver.

    The gate is applied over the whole ranked corpus, not a top-5 window: an
    on-topic chunk can rank low on lexical overlap with an abstract label (the
    heart-failure chunk ranks 17th of 30 for a "Circulatory" query). The
    keyword gate is what guarantees relevance, so widening the pool cannot
    admit an off-topic chunk.
    """
    keywords = _relevance_keywords(feature, value)
    if not keywords:
        return []
    candidates = retriever.search(_default_query(feature, value), k=len(retriever.chunks))
    return [c for c in candidates if _mentions(c["text"], keywords)][:k]


def _safe_patient_id(patient_id: object) -> str:
    """Reduce a caller-supplied identifier to a short token so it cannot carry
    instructions into the prompt."""
    return _PATIENT_ID_RE.sub("_", str(patient_id))[:_PATIENT_ID_MAX_LEN] or "unknown"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the training pipeline first (`make pipeline`).")
    with open(path) as f:
        return json.load(f)


@dataclass
class AgentResult:
    patient_id: str
    raw_risk: float
    calibrated_risk: float
    flagged_for_outreach: bool
    top_drivers: list[dict]
    retrieved_context: list[dict]
    narrative: str
    grounded: bool
    grounded_ratio: float
    fabricated_drug_mentions: list[str]
    unsupported_known_drivers: list[str]
    provider_used: str
    retry_count: int = 0
    extra: dict = field(default_factory=dict)


class ClinicalReasoningAgent:
    def __init__(self, provider: LLMProvider | None = None, retriever: Retriever | None = None):
        self.model = joblib.load(MODELS_DIR / "readmission_lgbm.joblib")
        self.calibrator = joblib.load(MODELS_DIR / "readmission_calibrator.joblib")
        self.schema = _load_json(MODELS_DIR / "feature_schema.json")
        meta = _load_json(MODELS_DIR / "model_meta.json")
        self.flag_threshold: float = meta["flag_threshold_raw_score"]
        self.base_rate: float = meta["base_rate_train"]

        self.explainer = shap.TreeExplainer(self.model)
        self.retriever = retriever or Retriever()
        self.provider = provider or get_provider()
        self.provider_name = type(self.provider).__name__

    def _build_feature_row(self, patient: dict) -> pd.DataFrame:
        """Coerce a raw patient record into the trained feature schema.

        Numeric fields are coerced (non-numeric -> NaN) and categorical values
        outside the training vocabulary are nulled before the Categorical is
        built. Besides keeping LightGBM's category codes consistent, this means
        free text placed in a structured field never reaches the prompt.
        """
        numeric_cols, categorical_cols = self.schema["numeric_cols"], self.schema["categorical_cols"]
        df = pd.DataFrame([{c: patient.get(c) for c in numeric_cols + categorical_cols}])
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in categorical_cols:
            allowed = self.schema["categories"][c]
            df[c] = pd.Categorical(df[c].where(df[c].isin(allowed), other=None), categories=allowed)
        return df[numeric_cols + categorical_cols]

    def _retrieve_for_drivers(self, drivers: list[dict]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
        retrieved_by_driver: dict[str, list[dict]] = {}
        dedup: dict[str, dict] = {}
        for d in drivers:
            kept = retrieve_evidence(self.retriever, d["feature"], d["value"])
            retrieved_by_driver[d["feature"]] = kept
            for hit in kept:
                dedup[hit["chunk_id"]] = hit
        return retrieved_by_driver, dedup

    def _build_prompt(
        self,
        patient_id: str,
        calibrated_risk: float,
        drivers: list[dict],
        retrieved_by_driver: dict[str, list[dict]],
    ) -> str:
        lines = [
            "You are a clinical decision-support assistant. Using ONLY the retrieved "
            "evidence below, write a short (4-6 sentence) explanation of this patient's "
            "30-day readmission risk and its main drivers. Cite the source (author, year) "
            "for every clinical claim. Do not recommend a specific drug, dose, or treatment "
            "change. If the evidence doesn't support a claim, say so instead of guessing.",
            f"\nPatient: {_safe_patient_id(patient_id)}",
            f"Calibrated 30-day readmission risk: {calibrated_risk:.1%} (cohort base rate: {self.base_rate:.1%})",
            "\nTop risk drivers for this patient and retrieved evidence:",
        ]
        for d in drivers:
            lines.append(f"\n- {d['feature']} = {d['value']}")
            for hit in retrieved_by_driver.get(d["feature"], []):
                lines.append(f'  * ({hit["authors"]}, {hit["year"]}) "{hit["text"]}"')
        return "\n".join(lines)

    def assess(self, patient: dict, patient_id: str = "unknown") -> AgentResult:
        patient_id = _safe_patient_id(patient_id)
        X = self._build_feature_row(patient)
        raw_risk = float(self.model.predict_proba(X)[:, 1][0])
        calibrated_risk = float(self.calibrator.predict([raw_risk])[0])

        drivers = top_risk_drivers(self.explainer, X, k=3)
        retrieved_by_driver, dedup = self._retrieve_for_drivers(drivers)
        all_hits = list(dedup.values())

        context = {
            "risk_score": calibrated_risk,
            "base_rate": self.base_rate,
            "drivers": drivers,
            "retrieved_by_driver": retrieved_by_driver,
        }
        prompt = self._build_prompt(patient_id, calibrated_risk, drivers, retrieved_by_driver)
        known_entities = {d["feature"] for d in drivers if d["feature"] in groundedness.KNOWN_DRUGS}

        narrative = self.provider.generate(prompt, context)
        report = groundedness.check(narrative, all_hits, known_entities=known_entities)
        retry_count = 0

        # Retrying only helps when generation drifted from the context. It is
        # a no-op for TemplateProvider, and it cannot fix a retrieval gap,
        # which is why a still-ungrounded result is returned flagged rather
        # than retried in a loop.
        if not report.grounded:
            stricter_prompt = (
                prompt + "\n\nYour previous answer included unsupported claims. Rewrite it "
                "using ONLY direct paraphrases of the quoted evidence above -- no outside "
                "knowledge, no speculation."
            )
            narrative = self.provider.generate(stricter_prompt, context)
            report = groundedness.check(narrative, all_hits, known_entities=known_entities)
            retry_count = 1

        return AgentResult(
            patient_id=patient_id,
            raw_risk=raw_risk,
            calibrated_risk=calibrated_risk,
            flagged_for_outreach=raw_risk >= self.flag_threshold,
            top_drivers=drivers,
            retrieved_context=all_hits,
            narrative=narrative,
            grounded=report.grounded,
            grounded_ratio=report.grounded_ratio,
            fabricated_drug_mentions=report.fabricated_drug_mentions,
            unsupported_known_drivers=report.unsupported_known_drivers,
            provider_used=self.provider_name,
            retry_count=retry_count,
        )


if __name__ == "__main__":
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    agent = ClinicalReasoningAgent()
    for _, row in test.sample(2, random_state=7).iterrows():
        result = agent.assess(row.to_dict(), patient_id=f"enc_{row['encounter_id']}")
        print(f"\n{'=' * 70}\nPatient {result.patient_id}")
        print(
            f"Calibrated risk: {result.calibrated_risk:.1%} | Flagged: {result.flagged_for_outreach} | "
            f"Grounded: {result.grounded} ({result.grounded_ratio:.0%}) | Provider: {result.provider_used}"
        )
        print(f"Top drivers: {[(d['feature'], d['value']) for d in result.top_drivers]}")
        print(f"\n{result.narrative}")
