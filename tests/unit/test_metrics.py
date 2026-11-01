from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from cliniq.metrics import brier_skill_score, capacity_threshold, cluster_bootstrap_ci, operating_points

TIED_SCORES = np.array([0.9] * 10 + [0.5] * 30 + [0.1] * 60)


def test_capacity_threshold_respects_capacity_with_ties():
    t = capacity_threshold(TIED_SCORES, 0.20)
    assert t == 0.9
    assert (TIED_SCORES >= t).mean() <= 0.20


def test_quantile_threshold_would_over_select_on_tied_scores():
    """The bug capacity_threshold exists to avoid."""
    naive = np.quantile(TIED_SCORES, 0.80)
    assert (TIED_SCORES >= naive).mean() == 0.40


def test_capacity_threshold_uses_full_capacity_when_cut_aligns():
    assert capacity_threshold(TIED_SCORES, 0.40) == 0.5


def test_capacity_threshold_never_exceeds_capacity_on_random_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        scores = rng.choice(np.linspace(0, 1, 25), size=500)
        rate = rng.uniform(0.05, 0.5)
        t = capacity_threshold(scores, rate)
        assert (scores >= t).mean() <= rate or t == scores.max()


def test_capacity_threshold_when_top_score_exceeds_capacity():
    scores = np.full(10, 0.3)
    assert capacity_threshold(scores, 0.2) == 0.3


def test_operating_points_with_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1, 0, 10)
    [op] = operating_points(y, scores, flag_rates=(0.2,))
    assert op["n_flagged"] == 2
    assert op["recall_at_flag_rate"] == 1.0
    assert op["precision_at_flag_rate"] == 1.0


def test_brier_skill_score_bounds():
    y = np.array([0, 0, 0, 1])
    assert math.isclose(brier_skill_score(y, np.full(4, 0.25)), 0.0, abs_tol=1e-12)
    assert brier_skill_score(y, y.astype(float)) == 1.0


def test_cluster_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(400), 2)
    y = rng.integers(0, 2, 800)
    scores = y * 0.5 + rng.normal(0, 0.5, 800)
    point = roc_auc_score(y, scores)
    ci = cluster_bootstrap_ci(y, scores, patients, roc_auc_score, n_boot=200)
    assert ci[0] < point < ci[1]
    assert ci == cluster_bootstrap_ci(y, scores, patients, roc_auc_score, n_boot=200)


def test_cluster_bootstrap_ci_is_nan_without_both_classes():
    y = np.zeros(20, dtype=int)
    lo, hi = cluster_bootstrap_ci(y, np.random.default_rng(0).random(20), np.arange(20), roc_auc_score, n_boot=50)
    assert math.isnan(lo) and math.isnan(hi)
