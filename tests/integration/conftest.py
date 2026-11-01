"""Integration tests need the processed data and trained model.

Missing artifacts skip these tests locally; set CLINIQ_REQUIRE_ARTIFACTS=1
(as CI does after running the pipeline) to make them fail instead.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from cliniq.config import CORPUS_DIR, MODELS_DIR, PROCESSED_DIR, ROOT

REQUIRED_ARTIFACTS = [
    PROCESSED_DIR / "train.parquet",
    PROCESSED_DIR / "test.parquet",
    MODELS_DIR / "readmission_lgbm.joblib",
    MODELS_DIR / "readmission_calibrator.joblib",
    MODELS_DIR / "feature_schema.json",
    MODELS_DIR / "model_meta.json",
    CORPUS_DIR / "chunks.jsonl",
]


@pytest.fixture(scope="session", autouse=True)
def _require_artifacts() -> None:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        message = f"pipeline artifacts missing (run `make pipeline`): {', '.join(missing)}"
        if os.environ.get("CLINIQ_REQUIRE_ARTIFACTS") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="session")
def test_split() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "test.parquet")


@pytest.fixture(scope="session")
def sample_patient(test_split: pd.DataFrame) -> dict:
    """A real test-set record; individual tests mutate copies of it."""
    return test_split.iloc[0].to_dict()
