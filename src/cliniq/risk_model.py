"""Train, calibrate, evaluate, and explain the 30-day readmission model.

1. LightGBM on the train split (native categorical handling), early stopping
   on val.
2. Isotonic calibration fit on val raw scores.
3. Outreach threshold chosen on val: the raw-score cut that flags at most
   DEFAULT_FLAG_RATE of patients.
4. Test-set evaluation: AUROC/AUPRC with patient-level bootstrap CIs, Brier
   score and Brier skill score, capacity-based operating points.
5. SHAP TreeExplainer for the global summary plot and per-patient drivers.

Usage:
    python -m cliniq.risk_model
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cliniq.config import DEFAULT_FLAG_RATE, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, TARGET
from cliniq.metrics import brier_skill_score, capacity_threshold, cluster_bootstrap_ci, operating_points

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

log = logging.getLogger(__name__)


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in ("train", "val", "test")}


def load_feature_lists() -> tuple[list[str], list[str]]:
    with open(PROCESSED_DIR / "prep_summary.json") as f:
        summary = json.load(f)
    return summary["numeric_features"], summary["categorical_features"]


def xy(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[numeric_cols + categorical_cols].copy()
    for c in categorical_cols:
        X[c] = X[c].astype("category")
    return X, df[TARGET]


def raw_scores(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability from the (uncalibrated) LightGBM model."""
    return np.asarray(model.predict_proba(X))[:, 1]


def train_model(
    train: pd.DataFrame, val: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> lgb.LGBMClassifier:
    X_train, y_train = xy(train, numeric_cols, categorical_cols)
    X_val, y_val = xy(val, numeric_cols, categorical_cols)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",  # ~11% positives; calibration restores probabilities
        random_state=SEED,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    return model


def logistic_baseline_auroc(
    train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> float:
    """Test AUROC of a one-hot logistic regression on the same features, to
    show how much the gradient-boosted model adds over a simple baseline."""

    def frame(df: pd.DataFrame) -> pd.DataFrame:
        return df[numeric_cols + categorical_cols].astype({c: object for c in categorical_cols})

    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore")
                ),
                categorical_cols,
            ),
        ]
    )
    baseline = make_pipeline(preprocess, LogisticRegression(max_iter=2000, class_weight="balanced"))
    baseline.fit(frame(train), train[TARGET])
    return float(roc_auc_score(test[TARGET], baseline.predict_proba(frame(test))[:, 1]))


def fit_calibrator(
    model: lgb.LGBMClassifier, val: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> IsotonicRegression:
    """Isotonic regression on val raw scores, fit directly rather than via
    CalibratedClassifierCV(cv="prefit"), whose prefit semantics have changed
    across scikit-learn releases."""
    X_val, y_val = xy(val, numeric_cols, categorical_cols)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores(model, X_val), y_val)
    return calibrator


def plot_calibration(y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, scores in (("Raw LightGBM", raw), ("Isotonic-calibrated", calibrated)):
        frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration: 30-day readmission risk (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shap_summary(explainer: shap.TreeExplainer, X_sample: pd.DataFrame, out_path: Path) -> None:
    shap_values = explainer.shap_values(X_sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig = plt.figure(figsize=(7, 6))
    shap.summary_plot(sv, X_sample, show=False, plot_size=None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_risk_drivers(explainer: shap.TreeExplainer, feature_row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Top-k features by SHAP contribution for one patient (largest push
    toward higher risk first). The agent turns these into retrieval queries."""
    shap_values = explainer.shap_values(feature_row)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = np.asarray(sv)[0]
    drivers = []
    for i in np.argsort(-sv)[:k]:
        value = feature_row.iloc[0, i]
        drivers.append(
            {
                "feature": str(feature_row.columns[i]),
                "value": value.item() if isinstance(value, np.generic) else value,
                "shap_contribution": float(sv[i]),
            }
        )
    return drivers


def evaluate(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, patients: np.ndarray) -> dict:
    """Discrimination and operating points are measured on the raw score (the
    ranking score); calibration on the calibrated probability."""
    return {
        "n_test_encounters": len(y),
        "n_test_patients": int(len(np.unique(patients))),
        "base_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, raw)),
        "auroc_ci95": list(cluster_bootstrap_ci(y, raw, patients, roc_auc_score)),
        "auprc": float(average_precision_score(y, raw)),
        "auprc_ci95": list(cluster_bootstrap_ci(y, raw, patients, average_precision_score)),
        "brier_raw": float(brier_score_loss(y, raw)),
        "brier_calibrated": float(brier_score_loss(y, calibrated)),
        "brier_skill_score": brier_skill_score(y, calibrated),
        "operating_points": operating_points(y, raw),
    }


def run() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols, categorical_cols = load_feature_lists()
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    model = train_model(train, val, numeric_cols, categorical_cols)
    calibrator = fit_calibrator(model, val, numeric_cols, categorical_cols)

    # The outreach cut is set on the raw score. Isotonic calibration is
    # monotone, so ranking is unchanged, but its output has only ~50 distinct
    # values: a cut on the calibrated score can only flag ~15% or ~26% here,
    # never the intended 20%. The raw score is continuous.
    X_val, _ = xy(val, numeric_cols, categorical_cols)
    val_raw = raw_scores(model, X_val)
    flag_threshold = capacity_threshold(val_raw, DEFAULT_FLAG_RATE)

    X_test, y_test = xy(test, numeric_cols, categorical_cols)
    raw = raw_scores(model, X_test)
    calibrated = calibrator.predict(raw)
    y = y_test.to_numpy()

    metrics = evaluate(y, raw, calibrated, test["patient_nbr"].to_numpy())
    metrics["baseline_logistic_regression_auroc"] = logistic_baseline_auroc(train, test, numeric_cols, categorical_cols)
    flagged = raw >= flag_threshold
    metrics["outreach_threshold"] = {
        "target_flag_rate": DEFAULT_FLAG_RATE,
        "raw_score_threshold": flag_threshold,
        "calibrated_risk_at_threshold": float(calibrator.predict([flag_threshold])[0]),
        "chosen_on": "validation",
        "realized_flag_rate_val": float((val_raw >= flag_threshold).mean()),
        "realized_flag_rate_test": float(flagged.mean()),
        "recall_test": float(flagged[y == 1].mean()),
        "precision_test": float(y[flagged].mean()),
    }

    plot_calibration(y, raw, calibrated, FIGURES_DIR / "calibration.png")
    explainer = shap.TreeExplainer(model)
    plot_shap_summary(
        explainer,
        X_test.sample(min(1000, len(X_test)), random_state=SEED),
        FIGURES_DIR / "shap_summary.png",
    )

    joblib.dump(model, MODELS_DIR / "readmission_lgbm.joblib")
    joblib.dump(calibrator, MODELS_DIR / "readmission_calibrator.joblib")

    # Persist each categorical column's exact category list. LightGBM encodes
    # pandas categoricals by integer code, so a single-row frame built at
    # inference time must be cast against these same categories.
    X_train, _ = xy(train, numeric_cols, categorical_cols)
    schema = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categories": {c: X_train[c].cat.categories.tolist() for c in categorical_cols},
    }
    with open(MODELS_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "best_iteration": int(model.best_iteration_),
        "flag_rate": DEFAULT_FLAG_RATE,
        "flag_threshold_raw_score": flag_threshold,
        "base_rate_train": float(train[TARGET].mean()),
        "versions": {"lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__, "shap": shap.__version__},
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "risk_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(
        "AUROC %.3f [%.3f, %.3f] | AUPRC %.3f | Brier skill %.3f | flag threshold %.4f (test flag rate %.1f%%)",
        metrics["auroc"],
        *metrics["auroc_ci95"],
        metrics["auprc"],
        metrics["brier_skill_score"],
        flag_threshold,
        100 * metrics["outreach_threshold"]["realized_flag_rate_test"],
    )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run()
