"""Consolidate every evaluation pillar into reports/eval_report.md.

Risk-model and fairness numbers are produced at training time
(cliniq.risk_model, cliniq.fairness_audit) and read from reports/*.json here.
Retrieval, generation, and the test suite run fresh.

Usage:
    python -m evals.run_all
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cliniq.config import REPORTS_DIR, ROOT
from evals.run_rag_evals import run as run_rag_evals
from evals.run_retrieval_eval import run as run_retrieval_eval


def run_test_suite() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "summary_line": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
        "output_tail": "\n".join(proc.stdout.splitlines()[-40:]),
    }


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ci(ci: list[float] | None) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def _risk_section(risk: dict | None) -> list[str]:
    lines = ["## 1. Risk model (test set)", ""]
    if not risk:
        return [*lines, "_reports/risk_model_metrics.json not found; run `make train`._"]
    ot = risk["outreach_threshold"]
    lines += [
        f"{risk['n_test_encounters']:,} encounters from {risk['n_test_patients']:,} patients; "
        f"base rate {_pct(risk['base_rate'])}. 95% CIs from a patient-level bootstrap.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AUROC | {risk['auroc']:.3f} {_ci(risk['auroc_ci95'])} |",
        f"| AUROC, logistic-regression baseline | {risk['baseline_logistic_regression_auroc']:.3f} |",
        f"| AUPRC | {risk['auprc']:.3f} {_ci(risk['auprc_ci95'])} |",
        f"| Brier score (calibrated) | {risk['brier_calibrated']:.4f} |",
        f"| Brier skill score vs. base rate | {risk['brier_skill_score']:.3f} |",
        "",
        "Capacity-based operating points (contact the top-k% by model score):",
        "",
        "| Flag rate | Patients flagged | Readmissions reached | Precision |",
        "|---|---|---|---|",
    ]
    for op in risk["operating_points"]:
        lines.append(
            f"| {_pct(op['flag_rate'])} | {op['n_flagged']:,} | "
            f"{_pct(op['recall_at_flag_rate'])} | {_pct(op['precision_at_flag_rate'])} |"
        )
    lines += [
        "",
        f"The deployed outreach rule (raw score >= {ot['raw_score_threshold']:.4f}, chosen on validation for a "
        f"{_pct(ot['target_flag_rate'])} capacity) flags {_pct(ot['realized_flag_rate_test'])} of test "
        f"patients, reaching {_pct(ot['recall_test'])} of readmissions at {_pct(ot['precision_test'])} precision.",
    ]
    return lines


def _fairness_section(fairness: dict | None) -> list[str]:
    lines = ["## 2. Fairness audit (test set, deployed threshold)", ""]
    if not fairness:
        return [*lines, "_reports/fairness_audit.json not found; run `make audit`._"]
    meta = fairness["_meta"]
    lines += [
        f"Disparity scalars include only groups with n >= {meta['min_group_n']} and "
        f">= {meta['min_group_positives']} readmissions.",
        "",
        "| Attribute | Demographic parity diff | Equalized odds diff | AUROC range (included groups) "
        "| Excluded (small sample) |",
        "|---|---|---|---|---|",
    ]
    for attr in ("race", "gender", "age_bucket"):
        s = fairness[attr]["summary"]
        excluded = ", ".join(s["groups_excluded_small_sample"]) or "none"
        lines.append(
            f"| {attr} | {s['demographic_parity_difference']:.3f} | {s['equalized_odds_difference']:.3f} "
            f"| {s['auroc_min']:.3f} - {s['auroc_max']:.3f} | {excluded} |"
        )
    lowest = min(
        (
            (attr, group, row)
            for attr in ("race", "gender", "age_bucket")
            for group, row in fairness[attr]["groups"].items()
            if row["included_in_summary"]
        ),
        key=lambda t: t[2]["auroc"],
    )
    attr, group, row = lowest
    lines += [
        "",
        f"Lowest-AUROC included group: {attr} = {group}, AUROC {row['auroc']:.3f} "
        f"{_ci(row['auroc_ci95'])} (n={row['n']:,}, {row['positives']} readmissions) vs. "
        f"{meta['overall_auroc']:.3f} overall.",
    ]
    return lines


def _retrieval_section(retrieval: dict) -> list[str]:
    meta = retrieval["_meta"]
    lines = [
        f"## 3. Retrieval ({meta['n_queries']}-query golden set, {meta['n_chunks']} chunks)",
        "",
        "| Method | MRR | Recall@3 | Recall@5 | Precision@3 |",
        "|---|---|---|---|---|",
    ]
    for method in ("bm25", "tfidf"):
        r = retrieval[method]
        lines.append(
            f"| {method} | {r['mrr']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['precision@3']:.3f} |"
        )
    return lines


def _generation_section(rag: dict) -> list[str]:
    return [
        f"## 4. Generation & guardrail ({rag['n_patients']} sampled test patients, provider: {rag['provider']})",
        "",
        f"Grounded: {_pct(rag['pct_grounded'])} | retried: {_pct(rag['pct_required_retry'])} | "
        f"fabricated drug mention: {_pct(rag['pct_with_fabricated_drug_mention'])} | "
        f"driver drug without corpus coverage: {_pct(rag['pct_with_unsupported_known_driver'])} | "
        f"heuristic rubric score: {rag['mean_rubric_score']:.2f}",
        "",
        "With TemplateProvider, narratives quote retrieved text verbatim, so groundedness and most "
        "rubric criteria pass by construction. These numbers verify the pipeline wiring; generation "
        "quality requires a generative provider (`CLINIQ_LLM_PROVIDER`).",
    ]


def build_markdown_report(risk: dict | None, fairness: dict | None, retrieval: dict, rag: dict, tests: dict) -> str:
    lines = ["# ClinIQ evaluation report", "", "Regenerate with `make eval`.", ""]
    lines += _risk_section(risk) + [""]
    lines += _fairness_section(fairness) + [""]
    lines += _retrieval_section(retrieval) + [""]
    lines += _generation_section(rag) + [""]
    lines += ["## 5. Test suite", "", f"`pytest tests`: {tests['summary_line'] or 'no output'}", ""]
    return "\n".join(lines)


def run() -> None:
    print("Running retrieval eval...")
    retrieval = run_retrieval_eval()
    print("Running generation eval...")
    rag = run_rag_evals()
    print("Running test suite...")
    tests = run_test_suite()
    print(f"  {tests['summary_line']}")

    report = build_markdown_report(
        _load_json(REPORTS_DIR / "risk_model_metrics.json"),
        _load_json(REPORTS_DIR / "fairness_audit.json"),
        retrieval,
        rag,
        tests,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval_report.md").write_text(report)
    print(f"Wrote {REPORTS_DIR / 'eval_report.md'}")

    if not tests["passed"]:
        print(tests["output_tail"])
        sys.exit(1)


if __name__ == "__main__":
    run()
