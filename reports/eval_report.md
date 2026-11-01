# ClinIQ evaluation report

Regenerate with `make eval`.

## 1. Risk model (test set)

14,867 encounters from 10,567 patients; base rate 11.0%. 95% CIs from a patient-level bootstrap.

| Metric | Value |
|---|---|
| AUROC | 0.667 [0.652, 0.682] |
| AUROC, logistic-regression baseline | 0.656 |
| AUPRC | 0.211 [0.189, 0.233] |
| Brier score (calibrated) | 0.0937 |
| Brier skill score vs. base rate | 0.041 |

Capacity-based operating points (contact the top-k% by model score):

| Flag rate | Patients flagged | Readmissions reached | Precision |
|---|---|---|---|
| 10.0% | 1,486 | 22.5% | 24.7% |
| 20.0% | 2,973 | 36.8% | 20.2% |
| 30.0% | 4,460 | 49.8% | 18.2% |

The deployed outreach rule (raw score >= 0.5808, chosen on validation for a 20.0% capacity) flags 18.7% of test patients, reaching 34.6% of readmissions at 20.3% precision.

## 2. Fairness audit (test set, deployed threshold)

Disparity scalars include only groups with n >= 100 and >= 20 readmissions.

| Attribute | Demographic parity diff | Equalized odds diff | AUROC range (included groups) | Excluded (small sample) |
|---|---|---|---|---|
| race | 0.106 | 0.195 | 0.634 - 0.738 | Asian |
| gender | 0.018 | 0.023 | 0.663 - 0.669 | Unknown/Invalid |
| age_bucket | 0.071 | 0.250 | 0.579 - 0.713 | [0-10), [10-20) |

Lowest-AUROC included group: age_bucket = [90-100), AUROC 0.579 [0.491, 0.663] (n=386, 52 readmissions) vs. 0.666 overall.

## 3. Retrieval (20-query golden set, 30 chunks)

| Method | MRR | Recall@3 | Recall@5 | Precision@3 |
|---|---|---|---|---|
| bm25 | 0.814 | 0.800 | 0.825 | 0.333 |
| tfidf | 0.797 | 0.800 | 0.825 | 0.333 |

## 4. Generation & guardrail (40 sampled test patients, provider: TemplateProvider)

Grounded: 100.0% | retried: 0.0% | fabricated drug mention: 0.0% | driver drug without corpus coverage: 10.0% | heuristic rubric score: 0.99

With TemplateProvider, narratives quote retrieved text verbatim, so groundedness and most rubric criteria pass by construction. These numbers verify the pipeline wiring; generation quality requires a generative provider (`CLINIQ_LLM_PROVIDER`).

## 5. Test suite

`pytest tests`: 90 passed, 3 xfailed in 4.68s
