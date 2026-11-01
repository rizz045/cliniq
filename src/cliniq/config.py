"""Project paths and run-time settings.

Paths resolve against the repository root. Set CLINIQ_ROOT to run the
pipeline against a different working directory (e.g. a mounted volume).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CLINIQ_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "readmitted_30d"
SEED = 42

# Share of discharges a care-management team can contact. The outreach
# threshold is chosen on the validation split so that at most this share of
# patients is flagged.
DEFAULT_FLAG_RATE = 0.20
