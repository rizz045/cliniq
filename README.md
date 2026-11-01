# ClinIQ

Explainable 30-day readmission risk for patients with diabetes. A calibrated
gradient-boosted model scores each discharge, SHAP identifies what drove the
score, and a retrieval layer explains those drivers with cited clinical
literature. A guardrail checks the narrative for unsupported claims before it
is returned.

```
patient record ──► LightGBM + isotonic calibration ──► risk score + outreach flag
                              │
                              ▼
                     SHAP top-3 drivers ──► relevance-gated retrieval (BM25, cited corpus)
                                                        │
                                                        ▼
                                          narrative (template or LLM) ──► groundedness check
                                                                              │
                                                     grounded=False ──► retry once, then flag for review
```

## Example output

```
Patient enc_207526128
Calibrated risk: 15.0% | Grounded: True (100%) | Provider: TemplateProvider

Estimated 30-day readmission risk: 15.0% (cohort base rate: 11.3%).

Contributing factor -- discharge_disposition_id_label = Discharged/transferred to SNF:
  No directly relevant passage was retrieved for this factor.

Contributing factor -- number_inpatient = 1:
  "A hospital discharge within the 90 days before the index admission was associated with
  76% higher odds of readmission (OR 1.76, 95% CI 1.64-1.89) ..." (Karunakaran A, Zhao H, Rubin DJ, 2018)

Contributing factor -- medical_specialty = Other:
  No directly relevant passage was retrieved for this factor.
```

When the corpus has no evidence for a driver, the agent says so rather than
citing something adjacent.

## Results

Test set: 14,867 encounters from 10,567 patients, never seen in training or
calibration. 95% confidence intervals come from a patient-level bootstrap.
The full report is in [`reports/eval_report.md`](reports/eval_report.md).

| Component | Metric | Result |
|---|---|---|
| Risk model | AUROC | 0.667 [0.652, 0.682] |
| | AUPRC (base rate 11.0%) | 0.211 [0.189, 0.233] |
| | Brier skill score vs. base-rate predictor | 0.041 |
| | Readmissions reached when contacting the top 20% | 36.8% at 20.2% precision |
| Fairness | Gender demographic parity difference | 0.018 |
| | AUROC, ages 50-60 vs. 80-90 | 0.711 [0.668, 0.750] vs. 0.609 [0.576, 0.644] |
| Retrieval | BM25 MRR / Recall@5 (20-query golden set) | 0.814 / 0.825 |
| Tests | `pytest` | 90 passed, 3 expected failures (known guardrail gaps) |

What the numbers say:

- **Moderate discrimination, typical of this dataset.** Structured
  encounter data carries limited signal for 30-day readmission, and models
  on this dataset commonly report AUROC in the mid-0.6s. The
  logistic-regression baseline trained alongside the model reaches 0.656,
  so gradient boosting adds about 0.01.
- **Discrimination degrades with age.** Patients aged 80-90 get
  significantly worse ranking than middle-aged patients (non-overlapping
  CIs), on large groups. Patients under 20 cannot be evaluated at all: 133
  test encounters, 3 readmissions.
- **Calibration is necessary, not a headline.** The class-weighted model's
  raw scores have a Brier score of 0.219. Calibration brings that to 0.094,
  but predicting the base rate for everyone already scores 0.098, so the
  Brier skill score (0.041) is the meaningful summary.
- **Generation metrics are a wiring check by default.** With the default
  template provider, narratives quote retrieved text verbatim, so
  groundedness is 100% by construction. A generative provider is needed to
  measure generation quality.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 if needed.

```bash
git clone <this-repo> && cd cliniq
make setup      # install dependencies + pre-commit hooks
make pipeline   # download data -> train -> fairness audit -> build corpus -> evals (~1-2 min)
make demo       # run the agent on two sample patients
```

`make pipeline` downloads the dataset from the UCI repository and verifies
its SHA-256 checksum. No data or model files are stored in this repo. Run
`make help` to list every target.

### Using a real LLM

The default provider is deterministic and needs no API key, so the
pipeline and CI run offline. To generate narratives with a model, set:

| Provider | Environment variables |
|---|---|
| Anthropic | `CLINIQ_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `claude-haiku-5-5`) |
| OpenAI | `CLINIQ_LLM_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_MODEL` |
| Ollama (local) | `CLINIQ_LLM_PROVIDER=ollama`, optional `OLLAMA_HOST`, `OLLAMA_MODEL` |

## Project layout

```
src/cliniq/
  config.py               paths and settings
  data_prep.py            download + verify data, clean, patient-level split
  risk_model.py           LightGBM, isotonic calibration, outreach threshold, SHAP
  metrics.py              bootstrap CIs, capacity threshold, operating points
  fairness_audit.py       per-group AUROC (with CIs) and outreach equity
  rag/                    corpus builder, BM25/TF-IDF retriever
  llm/providers.py        template / Anthropic / OpenAI / Ollama behind one interface
  guardrails/             groundedness and drug-name checks
  agent/clinical_agent.py orchestration and relevance gate
evals/                    retrieval eval, generation eval + rubric, report builder
tests/unit/               fast tests, no trained artifacts needed
tests/integration/        safety regressions against the trained model
data/corpus/sources.json  curated evidence corpus (7 sources, 30 passages)
reports/                  generated metrics, figures, eval_report.md
```

Design decisions and their trade-offs are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

```bash
make test        # all tests (integration tests skip until `make pipeline` has run)
make test-unit   # fast tests only
make lint        # ruff + format check + mypy
make format
```

CI (GitHub Actions) runs lint, type checks, and unit tests on every push. A
second job runs the full pipeline from a clean checkout and the integration
suite. Unit tests also enforce a retrieval-quality floor (BM25 MRR ≥ 0.75),
so a retrieval change that degrades results fails the build. A pre-commit
hook blocks data and model files from being committed.

## Limitations

- The groundedness check is lexical. It cannot detect negated or
  direction-flipped claims, skips very short sentences, and only recognises
  drugs present in the training data. Each gap is pinned by a strict
  `xfail` test in `tests/unit/test_groundedness.py`.
- Retrieval is lexical. Queries that share no vocabulary with their answer
  fail; one golden-set query ranks its answer 30th of 30. An experimental
  MedCPT dense retriever is included but not yet benchmarked.
- The corpus is small (7 sources, 30 passages) and hand-curated.
- All results are retrospective, on data from 1999-2008. There is no
  prospective validation and no drift monitoring.
- The validation split is used both for early stopping and for fitting the
  calibrator.

## Data and attribution

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008),
UCI Machine Learning Repository, CC BY 4.0. Strack B, DeShazo JP, Gennings C,
et al. "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis
of 70,000 Clinical Database Patient Records." *BioMed Research
International*, 2014. Corpus sources are cited individually in
[`data/corpus/sources.json`](data/corpus/sources.json).

## Disclaimer

A research and portfolio project, not a validated clinical tool. Do not use
it to inform clinical or care-management decisions.
