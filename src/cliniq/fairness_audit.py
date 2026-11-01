"""Disparate-impact audit across race, gender, and age band (test set).

race and gender are not model inputs, so this checks whether correlated
features reproduce group differences anyway. For each group it reports:

- AUROC with a patient-level bootstrap 95% CI (discrimination parity)
- selection rate, recall, and false-negative rate at the deployed outreach
  threshold (outreach equity)

Groups below MIN_GROUP_N encounters or MIN_GROUP_POSITIVES readmissions are
still reported, but excluded from the summary disparity scalars
(demographic parity / equalized odds difference). Those scalars are a
max-minus-min across groups, so one tiny group can dominate them: on this
data a single "Unknown/Invalid" gender record (n=1) inflates the gender
demographic parity difference from about 0.02 to about 0.20.

Usage:
    python -m cliniq.fairness_audit
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import roc_auc_score

from cliniq.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET
from cliniq.metrics import cluster_bootstrap_ci

log = logging.getLogger(__name__)

MIN_GROUP_N = 100
MIN_GROUP_POSITIVES = 20
AUDITED_ATTRIBUTES = ("race", "gender", "age_bucket")


def _group_table(
    y: np.ndarray, scores: np.ndarray, flagged: np.ndarray, groups: np.ndarray, patients: np.ndarray
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for group in sorted(np.unique(groups)):
        m = groups == group
        n, positives = int(m.sum()), int(y[m].sum())
        has_both_classes = 0 < positives < n
        recall = float(flagged[m][y[m] == 1].mean()) if positives else None
        table[str(group)] = {
            "n": n,
            "positives": positives,
            "base_rate": positives / n,
            "auroc": float(roc_auc_score(y[m], scores[m])) if has_both_classes else None,
            "auroc_ci95": (
                list(cluster_bootstrap_ci(y[m], scores[m], patients[m], roc_auc_score)) if has_both_classes else None
            ),
            "selection_rate": float(flagged[m].mean()),
            "recall": recall,
            "false_negative_rate": None if recall is None else 1.0 - recall,
            "included_in_summary": n >= MIN_GROUP_N and positives >= MIN_GROUP_POSITIVES,
        }
    return table


def audit_attribute(
    y: np.ndarray, scores: np.ndarray, flagged: np.ndarray, groups: np.ndarray, patients: np.ndarray
) -> dict:
    table = _group_table(y, scores, flagged, groups, patients)
    included = [g for g, row in table.items() if row["included_in_summary"]]
    mask = np.isin(groups, included)
    aurocs = [table[g]["auroc"] for g in included]
    return {
        "groups": table,
        "summary": {
            "groups_included": included,
            "groups_excluded_small_sample": [g for g in table if g not in included],
            "demographic_parity_difference": float(
                demographic_parity_difference(y[mask], flagged[mask], sensitive_features=groups[mask])
            ),
            "equalized_odds_difference": float(
                equalized_odds_difference(y[mask], flagged[mask], sensitive_features=groups[mask])
            ),
            "auroc_min": min(aurocs),
            "auroc_max": max(aurocs),
        },
    }


def run() -> dict:
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    model = joblib.load(MODELS_DIR / "readmission_lgbm.joblib")
    calibrator = joblib.load(MODELS_DIR / "readmission_calibrator.joblib")
    with open(MODELS_DIR / "feature_schema.json") as f:
        schema = json.load(f)
    with open(MODELS_DIR / "model_meta.json") as f:
        threshold = json.load(f)["flag_threshold_raw_score"]

    X = test[schema["numeric_cols"] + schema["categorical_cols"]].copy()
    for c in schema["categorical_cols"]:
        X[c] = X[c].astype("category")
    y = test[TARGET].to_numpy()
    raw = model.predict_proba(X)[:, 1]
    scores = calibrator.predict(raw)
    flagged = (raw >= threshold).astype(int)
    patients = test["patient_nbr"].to_numpy()

    attributes = {
        "race": test["race"].fillna("Unknown").astype(str).to_numpy(),
        "gender": test["gender"].fillna("Unknown").astype(str).to_numpy(),
        "age_bucket": test["age"].astype(str).to_numpy(),
    }
    results: dict[str, dict] = {
        name: audit_attribute(y, scores, flagged, groups, patients) for name, groups in attributes.items()
    }
    results["_meta"] = {
        "overall_auroc": float(roc_auc_score(y, scores)),
        "flag_threshold_raw_score": threshold,
        "realized_flag_rate": float(flagged.mean()),
        "min_group_n": MIN_GROUP_N,
        "min_group_positives": MIN_GROUP_POSITIVES,
        "ci_method": "patient-level cluster bootstrap, 500 replicates, percentile 95% CI",
        "note": (
            "race and gender are not model inputs; this audit measures disparate impact via "
            "correlated features. Disparity scalars are in [0, 1]; 0 means no gap between "
            "the best- and worst-off included group."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "fairness_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    for name in AUDITED_ATTRIBUTES:
        s = results[name]["summary"]
        log.info(
            "%s: DPD %.3f | EOD %.3f | AUROC range [%.3f, %.3f] | excluded (small n): %s",
            name,
            s["demographic_parity_difference"],
            s["equalized_odds_difference"],
            s["auroc_min"],
            s["auroc_max"],
            s["groups_excluded_small_sample"],
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run()
