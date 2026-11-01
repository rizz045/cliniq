"""Evaluation helpers shared by the risk model and the fairness audit."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import brier_score_loss

from cliniq.config import SEED


def capacity_threshold(scores: np.ndarray, flag_rate: float) -> float:
    """Lowest score cut ``t`` such that flagging ``scores >= t`` selects at
    most ``flag_rate`` of patients.

    With tied scores, a plain quantile threshold applied with ``>=`` pulls in
    every tie at the boundary and over-selects; on this model's isotonic
    output it flagged 25.7% instead of 20%. This picks the cut that respects
    capacity instead. If even the top score exceeds capacity, the top score is
    returned.
    """
    scores = np.asarray(scores, dtype=float)
    values = np.unique(scores)[::-1]  # distinct scores, high -> low
    ascending_neg = np.sort(-scores)
    n_at_or_above = np.searchsorted(ascending_neg, -values, side="right")
    shares = n_at_or_above / len(scores)
    within_capacity = values[shares <= flag_rate]
    return float(within_capacity[-1]) if within_capacity.size else float(values[0])


def operating_points(
    y_true: np.ndarray, scores: np.ndarray, flag_rates: tuple[float, ...] = (0.10, 0.20, 0.30)
) -> list[dict]:
    """Recall and precision when contacting the top-k% of patients by score.

    Framed around a capacity-constrained outreach program: if a team can call
    the top 20% highest-risk discharges, what share of actual readmissions
    does that reach?
    """
    y_true = np.asarray(y_true)
    n = len(scores)
    order = np.argsort(-np.asarray(scores), kind="stable")
    points = []
    for rate in flag_rates:
        k = max(1, int(n * rate))
        caught = int(y_true[order[:k]].sum())
        points.append(
            {
                "flag_rate": rate,
                "n_flagged": k,
                "readmissions_caught": caught,
                "recall_at_flag_rate": caught / float(y_true.sum()),
                "precision_at_flag_rate": caught / k,
            }
        )
    return points


def brier_skill_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Brier skill relative to predicting the observed base rate for everyone.

    0 means no better than the base rate, 1 is perfect. Reported alongside the
    raw Brier score because at an ~11% base rate a constant predictor already
    scores ~0.098, so a low Brier score alone says little.
    """
    y_true = np.asarray(y_true, dtype=float)
    reference = brier_score_loss(y_true, np.full_like(y_true, y_true.mean()))
    return float(1.0 - brier_score_loss(y_true, probs) / reference)


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    metric: Callable[..., float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile confidence interval from a patient-level (cluster) bootstrap.

    Encounters from the same patient are correlated, so resampling rows would
    understate the variance. Each replicate draws patients with replacement
    and weights every encounter by its patient's draw count. ``metric`` must
    accept ``sample_weight`` (roc_auc_score, average_precision_score, ...).
    Returns (nan, nan) when too few replicates contain both classes.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    uniques, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = len(uniques)
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(n_boot):
        draw_counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        weights = draw_counts[codes]
        keep = weights > 0
        y_rep = y_true[keep]
        if y_rep.min() == y_rep.max():
            continue
        stats.append(metric(y_rep, scores[keep], sample_weight=weights[keep]))

    if len(stats) < n_boot // 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
