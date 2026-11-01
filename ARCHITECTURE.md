# Architecture and design decisions

ClinIQ has two layers. The **risk model** has to be right: a calibrated,
explainable, fairness-audited classifier that ranks discharges for
outreach. The **reasoning layer** makes the score usable: it turns the
model's top drivers into a short narrative backed by cited evidence, without
inventing clinical claims. Most of the interesting engineering sits at the
seam between them: SHAP driver → retrieval query → cited narrative →
groundedness check.

## Contents

1. [System overview](#system-overview)
2. [Data](#data)
3. [Risk model](#risk-model)
4. [Fairness audit](#fairness-audit)
5. [Evidence corpus and retrieval](#evidence-corpus-and-retrieval)
6. [Agent and relevance gate](#agent-and-relevance-gate)
7. [Groundedness guardrail](#groundedness-guardrail)
8. [LLM providers](#llm-providers)
9. [Evaluation and testing](#evaluation-and-testing)
10. [Limitations and next steps](#limitations-and-next-steps)

## System overview

```
UCI dataset 296 (101,766 encounters)
        │  cliniq.data_prep: download + checksum, clean, patient-level split
        ▼
train / val / test parquet ──► cliniq.risk_model: LightGBM → isotonic calibration
                                    │       → outreach threshold (val) → SHAP
                                    ▼
                         models/ (model, calibrator, schema, metadata)
                                    │
             ┌──────────────────────┴───────────────────────┐
             ▼                                              ▼
 cliniq.fairness_audit                       cliniq.agent.ClinicalReasoningAgent.assess()
 per-group AUROC + CIs,                        1. predict + calibrate
 outreach equity                               2. top-3 SHAP drivers
                                               3. retrieve evidence per driver (relevance gate)
                                               4. generate narrative (LLMProvider)
                                               5. groundedness check → retry once → flag
```

`assess()` runs per patient at request time. Training, calibration, and the
audit run offline.

## Data

**Diabetes 130-US Hospitals, 1999-2008** (UCI 296; Strack et al., 2014; CC
BY 4.0): 101,766 inpatient encounters from 70,439 patients across 130 US
hospitals. The label is readmission within 30 days (`readmitted == "<30"`).

`data_prep.py` downloads the official UCI archive and verifies SHA-256
checksums of both files before use. Decisions:

- **Patient-level split (70/15/15).** 23% of test patients have more than one
  encounter. Splitting by encounter would put the same patient in train and
  test and inflate results.
- **race and gender are excluded from features** but kept for the audit.
  This guards against disparate treatment, not disparate impact, which is
  what the audit measures. Age enters only as a numeric midpoint.
- **weight (~97% missing) and payer_code are dropped.** payer_code is also an
  insurance-status proxy that should not drive a risk score.
- **Expired and hospice discharges are removed** (1,652 encounters):
  readmission is undefined for them.
- **ICD-9 codes are collapsed into the Strack et al. diagnosis groups** rather
  than ~700 sparse raw codes. medical_specialty keeps its top 15 values.
- **Categoricals stay pandas `category`** for LightGBM's native handling, and
  the exact category lists are persisted so single-row inference frames get
  the same integer codes as training.

Result: 100,114 encounters, 30 features (9 numeric, 21 categorical). Splits:
69,968 / 15,279 / 14,867 with readmission rates 11.3% / 11.9% / 11.0%.

## Risk model

LightGBM (`class_weight="balanced"`, early stopping on validation AUROC),
then isotonic regression fit on the validation split's raw scores. Isotonic
is fit directly rather than through `CalibratedClassifierCV(cv="prefit")`,
whose prefit semantics have changed across scikit-learn versions.

Test results (95% CIs from a patient-level cluster bootstrap, 500
replicates):

| Metric | Value |
|---|---|
| AUROC | 0.667 [0.652, 0.682] |
| AUPRC (base rate 11.0%) | 0.211 [0.189, 0.233] |
| Logistic-regression baseline AUROC | 0.656 |
| Brier score, raw → calibrated | 0.219 → 0.094 |
| Brier skill score vs. base-rate predictor | 0.041 |

Two of these need context:

- **Brier score.** The raw 0.219 is inflated by class weighting. Predicting
  the 11% base rate for everyone scores 0.098, so the useful figure is the
  skill score: the calibrated model is about 4% better than that
  reference. Calibration is still necessary, because it is what makes "15%"
  mean 15%. It is not, by itself, evidence of predictive power.
- **Bootstrap by patient.** Encounters from the same patient are correlated,
  so CIs resample patients and weight their encounters, rather than
  resampling rows.

### Outreach threshold

A care-management team can contact only a fixed share of discharges, so the
model is evaluated by what it catches at a capacity, not by accuracy at
0.5:

| Contact top | Patients | Readmissions reached | Precision |
|---|---|---|---|
| 10% | 1,486 | 22.5% | 24.7% |
| 20% | 2,973 | 36.8% | 20.2% |
| 30% | 4,460 | 49.8% | 18.2% |

The deployed flag needs one threshold, chosen on validation data, not test.
The obvious implementation, `np.quantile(calibrated, 0.8)` applied with `>=`,
flags **25.7%** of patients. Isotonic output is a step function with about
50 distinct values, so every patient tied at the boundary step is swept in.
A capacity-respecting cut on the same calibrated scores swings the other
way and flags only 15.2%.

The fix is to rank on the raw model score. Isotonic calibration is
monotone, so the ranking is identical, but the raw score is continuous. The
cut flags 20.0% on validation and 18.7% on test, reaching 34.6% of test
readmissions at 20.3% precision. The calibrated probability is still what is
reported to users. The same reasoning is why AUROC and AUPRC are computed on
the raw score: tied calibrated scores slightly understate ranking quality.

### Explanations

One `shap.TreeExplainer` produces both the global summary plot
(`reports/figures/shap_summary.png`) and each patient's top-3 drivers, so a
per-patient explanation uses the same mechanism a model reviewer inspects.

## Fairness audit

`fairness_audit.py` measures, per race, gender, and age band on the test
set: AUROC with a patient-level bootstrap CI, plus selection rate, recall,
and false-negative rate at the deployed threshold. Summary scalars
(fairlearn's demographic parity and equalized odds differences) use only
groups with at least 100 encounters and 20 readmissions.

That minimum matters. The scalars are max-minus-min across groups, so one
tiny group dominates them. A single `Unknown/Invalid` gender record (n=1,
selection rate 0) moves the gender demographic parity difference from 0.018
to 0.196. Groups below the minimum are still reported, with CIs, just not
folded into the summary.

Findings:

| Attribute | Demographic parity diff | Equalized odds diff | AUROC range (included groups) |
|---|---|---|---|
| race | 0.106 | 0.195 | 0.634 - 0.738 |
| gender | 0.018 | 0.023 | 0.663 - 0.669 |
| age band | 0.071 | 0.250 | 0.579 - 0.713 |

- **Age is the clearest effect.** AUROC falls from 0.711 [0.668, 0.750] for
  ages 50-60 to 0.609 [0.576, 0.644] for 80-90 (n=2,408), and 0.579 [0.491,
  0.663] for 90-100. The intervals for 50-60 and 80-90 do not overlap.
  Ranking quality is worst for the oldest patients, who have the highest
  readmission rates.
- **Under-20s cannot be evaluated.** The 0-10 and 10-20 bands have 133 test
  encounters and 3 readmissions between them. Their point AUROC (0.281) is
  meaningless at that size. The right response is a scope rule ("not
  validated for patients under 20"), not a fairness correction.
- **Gender shows no material gap.** Race shows wider AUROC spread, mostly in
  smaller groups with wide intervals.

## Evidence corpus and retrieval

`data/corpus/sources.json` holds 7 sources (CMS HRRP policy, five
peer-reviewed studies, one meta-analysis), distilled into 30 single-claim
passages. Each passage carries author, year, and URL, so every sentence the
agent writes can be attributed. A small curated corpus was chosen over a
bulk crawl: for clinical grounding, provenance per passage matters more than
volume.

| source_id | First author, year | Venue |
|---|---|---|
| cms_hrrp | CMS, 2024 | HRRP policy page |
| karunakaran_2018 | Karunakaran A, 2018 | Medical Care |
| wei_2013 | Wei NJ, 2013 | Diabetic Medicine |
| drc_2016_e000160 | Emons MF, 2016 | BMJ Open Diabetes Research & Care |
| drc_meta_2020_e001227 | Soh JGS, 2020 | BMJ Open Diabetes Research & Care (meta-analysis) |
| singapore_2023 | Soh JGS, 2023 | Scientific Reports |
| engagement_2013 | Ronksley PE, 2013 | BMC Health Services Research |

**Retrieval** is BM25 (`rank_bm25`), with TF-IDF cosine as a baseline.
Tokenisation lowercases text, drops stopwords, and strips a trailing plural
"s". On a 20-query hand-labeled golden set: BM25 MRR 0.814, Recall@3 0.800,
Recall@5 0.825; TF-IDF MRR 0.797.

What the golden set showed:

- **Stopword filtering is essential** on single-sentence chunks, where an
  incidental "is" or "to" can outweigh one real content-word match.
- **The plural stripper costs MRR** (0.848 without it, 0.814 with it). It
  turns "comorbidities" into "comorbiditie", which no longer matches
  "comorbidity". It is kept as a bet on a larger corpus but should be
  replaced by a real lemmatiser or removed.
- **Vocabulary mismatch defeats lexical retrieval.** "Which medical
  conditions are tracked by the hospital readmissions reduction program"
  ranks its answer 30th of 30: the passage lists condition names and shares
  no terms with the question. That is the concrete case for dense
  retrieval. `DenseRetriever` (MedCPT) implements the same interface but is
  not yet benchmarked.

## Agent and relevance gate

`ClinicalReasoningAgent.assess()` is a plain, explicit call sequence rather
than an agent framework: the steps are fixed, so the control flow should be
readable top to bottom. A framework would earn its place if the LLM had to
choose the next action, for example deciding whether more retrieval is
needed.

### A failure the groundedness check cannot catch

The groundedness check verifies that the narrative matches the retrieved
text. It cannot tell whether the retrieved text was right for the patient.
Two real cases:

1. A patient **discharged to a skilled nursing facility** was paired with
   evidence about patients who **left against medical advice**. Both are
   "discharge" passages, so BM25 ranked the AMA passage highly. The
   narrative quoted it faithfully and scored as fully grounded, but it was
   the wrong evidence.
2. A **Circulatory** diagnosis driver was explained with an age statistic.
   The relevant passage (heart failure as a top comorbidity predictor)
   ranked 17th of 30 for a query built from the bucket label, because
   "Circulatory" never appears in it.

**The fix is a per-feature relevance gate** (`retrieve_evidence`). A chunk
counts as evidence for a driver only if it contains a feature-specific
keyword, matched at word start, so that "age" does not match "percentage".
For discharge disposition and diagnosis group, the keywords depend on the
value: "Left AMA" requires AMA language, and "Circulatory" requires "heart
failure" or "cardiovascular". The gate fails closed. A feature or value with
no keyword list returns no evidence, and the narrative states that nothing
relevant was found. For gated features, the gate scans the whole ranked
corpus rather than a top-5 window: the keyword match guarantees relevance,
so widening the pool can only surface the right chunk, never admit a wrong
one.

The broader lesson: an always-green grounding metric can coexist with wrong
answers. Reading real outputs against real patients found these cases, and
each is now a regression test.

### Input handling

A patient record comes from upstream systems, so it is treated as
untrusted. Numeric fields are coerced, and non-numeric values become
missing. Categorical values outside the training vocabulary are nulled
before encoding. The caller-supplied patient ID is reduced to
`[A-Za-z0-9._-]{1,64}`. No free text from the record can reach the prompt.

### Retry semantics

On a failed check, the agent regenerates once with a stricter prompt and
then returns the result with `grounded=False`, for a caller to route to
human review. A retry can fix generation drift. It cannot fix a retrieval
gap, which is why the result is flagged rather than retried in a loop.

## Groundedness guardrail

`guardrails/groundedness.py` is deterministic and runs on every call. It is
deliberately not a second LLM call, which could fail the same way the first
did.

1. **Claim overlap.** Every sentence with at least 4 content tokens must
   share at least 40% of its tokens with some retrieved chunk. At least 80%
   of sentences must pass.
2. **Drug-name leakage.** A known diabetes drug named in the narrative but
   absent from the retrieved context is a fabrication, unless it is one of
   the patient's own SHAP drivers. In that case it is reported as a
   corpus-coverage gap.

Template boilerplate (the "Contributing factor -- X = Y" labels and the
disclaimer) is excluded from scoring, since a label sharing no vocabulary
with the corpus is not an unsupported claim.

**Known gaps**, each pinned by a strict `xfail` test so a stronger checker
will flip it to passing:

- negated or direction-flipped claims ("was **not** associated") score as
  grounded, because token overlap ignores polarity;
- drugs outside the training data (e.g. SGLT2 inhibitors) are not
  recognised;
- sentences under 4 content tokens are not scored, so a short directive
  ("Stop insulin now.") passes.

The intended upgrade is structured output: the LLM returns sentence →
chunk_id citations, and each pair is checked with an entailment model.

## LLM providers

`llm/providers.py` defines one interface, `generate(prompt, context) ->
str`, with four implementations selected by `CLINIQ_LLM_PROVIDER`:
template (default), Anthropic, OpenAI, and Ollama. The agent depends only on
the interface.

The default `TemplateProvider` renders the narrative from structured
context: numbers from the model, sentences quoted verbatim from retrieved
chunks. It cannot produce an unsupported claim, which makes it a
reproducible baseline. CI runs offline, and when its output fails the
groundedness check, the cause is necessarily retrieval rather than
generation. The trade-off is that generation metrics measured with it are
wiring checks, not quality measurements.

## Evaluation and testing

| Pillar | Where | What |
|---|---|---|
| Risk model | `cliniq.risk_model` | AUROC/AUPRC with CIs, Brier skill, operating points, LR baseline |
| Fairness | `cliniq.fairness_audit` | per-group AUROC with CIs, outreach equity |
| Retrieval | `evals/run_retrieval_eval.py` | MRR, Recall@k, Precision@k on the golden set |
| Generation | `evals/run_rag_evals.py` | groundedness + 5-criterion weighted rubric on 40 test patients |
| Report | `evals/run_all.py` | consolidates everything into `reports/eval_report.md` |

The rubric follows the HealthBench pattern: weighted pass/fail criteria
covering citations, no dosing advice, risk stated against the base rate, a
disclaimer, and every driver addressed. It has a heuristic grader and an
LLM-judge grader. Under the template provider, most criteria pass by
construction.

**Tests.** `tests/unit` (66 tests, 3 of them strict xfails) needs no trained artifacts and covers data
cleaning, metrics, retrieval, the guardrail, the agent helpers, providers,
and the rubric. It also enforces a retrieval floor (BM25 MRR ≥ 0.75,
Recall@5 ≥ 0.75), so a retrieval change that degrades quality fails CI.
`tests/integration` (27 tests) runs against the trained model: injection resistance,
the relevance-gate regressions, retry flow, sensitive-attribute exclusion,
schema/model consistency, and the outreach-threshold capacity. Locally,
integration tests skip if artifacts are missing. CI sets
`CLINIQ_REQUIRE_ARTIFACTS=1` after running the pipeline, so they fail
instead.

## Limitations and next steps

- **Evidence quality.** Replace lexical overlap with structured citations
  plus entailment checking, and expand the drug lexicon (RxNorm).
- **Retrieval.** Benchmark the MedCPT dense retriever and hybrid BM25+dense
  ranking (reciprocal rank fusion) on an expanded golden set.
- **Generation evaluation.** Run the rubric against real LLM providers, and
  validate the LLM judge against human labels.
- **Serving.** Add a FastAPI service with typed request schemas, a review
  queue for ungrounded narratives, and a scope rule for under-20 patients.
- **Monitoring.** Track feature and score drift, calibration over time, and
  the rate of ungrounded narratives.
- **Validation.** All results are retrospective on 1999-2008 data. The
  validation split serves both early stopping and calibration; a separate
  calibration split would remove that coupling.
