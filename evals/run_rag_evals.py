"""End-to-end generation eval: run the full agent on a fixed sample of
test-set patients and score each narrative with the groundedness guardrail
and the heuristic rubric.

Usage:
    python -m evals.run_rag_evals
"""

from __future__ import annotations

import json

import pandas as pd

from cliniq.agent.clinical_agent import ClinicalReasoningAgent
from cliniq.config import PROCESSED_DIR, REPORTS_DIR
from evals.rubrics import grade_heuristic

N_SAMPLES = 40
SEED = 13


def _rate(records: list[dict], predicate) -> float:  # type: ignore[no-untyped-def]
    return round(sum(bool(predicate(r)) for r in records) / len(records), 3)


def run(agent: ClinicalReasoningAgent | None = None) -> dict:
    agent = agent or ClinicalReasoningAgent()
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    sample = test.sample(min(N_SAMPLES, len(test)), random_state=SEED)

    records = []
    for _, row in sample.iterrows():
        result = agent.assess(row.to_dict(), patient_id=f"enc_{row['encounter_id']}")
        rubric = grade_heuristic(result.narrative, {"drivers": result.top_drivers, "base_rate": agent.base_rate})
        records.append(
            {
                "patient_id": result.patient_id,
                "calibrated_risk": result.calibrated_risk,
                "flagged_for_outreach": result.flagged_for_outreach,
                "grounded": result.grounded,
                "grounded_ratio": result.grounded_ratio,
                "fabricated_drug_mentions": result.fabricated_drug_mentions,
                "unsupported_known_drivers": result.unsupported_known_drivers,
                "retry_count": result.retry_count,
                "rubric_score": rubric["score"],
                "rubric_criteria": rubric["criteria"],
            }
        )

    n = len(records)
    criterion_ids = [c["criterion_id"] for c in records[0]["rubric_criteria"]]
    per_criterion = {
        cid: _rate(
            records, lambda r, cid=cid: next(c["passed"] for c in r["rubric_criteria"] if c["criterion_id"] == cid)
        )
        for cid in criterion_ids
    }
    summary = {
        "provider": agent.provider_name,
        "n_patients": n,
        "pct_grounded": _rate(records, lambda r: r["grounded"]),
        "mean_grounded_ratio": round(sum(r["grounded_ratio"] for r in records) / n, 3),
        "pct_required_retry": _rate(records, lambda r: r["retry_count"] > 0),
        "pct_with_fabricated_drug_mention": _rate(records, lambda r: r["fabricated_drug_mentions"]),
        "pct_with_unsupported_known_driver": _rate(records, lambda r: r["unsupported_known_drivers"]),
        "mean_rubric_score": round(sum(r["rubric_score"] for r in records) / n, 3),
        "per_criterion_pass_rate": per_criterion,
        "pct_flagged_for_outreach": _rate(records, lambda r: r["flagged_for_outreach"]),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "rag_eval.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
