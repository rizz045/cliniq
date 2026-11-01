"""Load, clean, and split the UCI Diabetes 130-US Hospitals dataset (ID 296).

Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates",
BioMed Research International, 2014. CC BY 4.0.

Key decisions (rationale in ARCHITECTURE.md):

1. Split by patient (``patient_nbr``), not encounter, so no patient appears in
   more than one of train/val/test.
2. race and gender are kept for the fairness audit but excluded from model
   features; age enters only as a numeric midpoint.
3. weight (~97% missing) and payer_code (sparse, insurance-status proxy) are
   dropped.
4. ICD-9 diagnosis codes are collapsed into the Strack et al. groups instead
   of ~700 raw codes.
5. Categoricals stay pandas ``category`` dtype for LightGBM's native handling.

Usage:
    python -m cliniq.data_prep
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cliniq.config import PROCESSED_DIR, RAW_DIR, SEED, TARGET

log = logging.getLogger(__name__)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
# SHA-256 of the files in the official UCI release. The download is rejected
# if either file differs.
RAW_FILES = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDs_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}

EXPIRED_DISCHARGE_IDS = {11, 19, 20, 21}  # died or hospice: readmission is undefined

SENSITIVE_COLS = ["race", "gender", "age"]  # kept for the fairness audit only
DROP_COLS = ["weight", "payer_code"]  # too sparse / insurance-status proxy
ID_COLS = ["encounter_id", "patient_nbr"]
RAW_ONLY_COLS = [
    "readmitted",
    "diag_1",
    "diag_2",
    "diag_3",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]
MAPPED_ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

_AGE_MIDPOINT = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_raw_data(raw_dir: Path = RAW_DIR) -> None:
    """Download the official UCI release if needed, then verify checksums."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(not (raw_dir / name).exists() for name in RAW_FILES):
        log.info("Downloading UCI dataset 296 from %s", UCI_ZIP_URL)
        resp = requests.get(UCI_ZIP_URL, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # The archive spells it "IDS_mapping.csv"; match case-insensitively.
            members = {Path(m).name.lower(): m for m in zf.namelist()}
            for name in RAW_FILES:
                (raw_dir / name).write_bytes(zf.read(members[name.lower()]))

    for name, expected in RAW_FILES.items():
        actual = _sha256(raw_dir / name)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {name} (expected {expected}, got {actual}). "
                f"Delete {raw_dir} and re-run to download a fresh copy."
            )


def group_icd9(code: object) -> str:
    """Map an ICD-9 code to the Strack et al. (2014) diagnosis group."""
    if not isinstance(code, str) or code in ("?", ""):
        return "Missing"
    if code.startswith("250"):
        return "Diabetes"
    try:
        val = float(code)
    except ValueError:  # V- and E- codes (supplementary / external causes)
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def load_id_mappings(path: Path) -> dict[str, pd.DataFrame]:
    """Parse IDs_mapping.csv, which stacks three lookup tables in one file.

    Blocks are separated by a bare "," line rather than a blank line, so
    boundaries are detected by header text.
    """
    mappings: dict[str, pd.DataFrame] = {}
    current_header: str | None = None
    current_rows: list[list[str]] = []

    def _flush() -> None:
        if current_header and current_rows:
            df = pd.DataFrame(current_rows, columns=[current_header, "description"])
            df[current_header] = pd.to_numeric(df[current_header], errors="coerce")
            df["description"] = df["description"].str.strip().str.strip('"')
            mappings[current_header] = df.dropna(subset=[current_header])

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line == ",":
            continue
        head, _, rest = line.partition(",")
        if head in MAPPED_ID_COLS and rest == "description":
            _flush()
            current_header, current_rows = head, []
            continue
        if current_header:
            current_rows.append(line.split(",", 1))
    _flush()
    return mappings


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "diabetic_data.csv", low_memory=False)


def clean(df: pd.DataFrame, id_mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)].copy()
    log.info("Dropped %d encounters with an expired/hospice discharge.", before - len(df))

    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    df["age_midpoint"] = df["age"].map(_AGE_MIDPOINT)
    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_group"] = df[col].map(group_icd9)

    specialty = df["medical_specialty"].fillna("Unknown")
    top_specialties = specialty.value_counts().nlargest(15).index
    df["medical_specialty"] = specialty.where(specialty.isin(top_specialties), "Other")

    for id_col in MAPPED_ID_COLS:
        mapping = id_mappings[id_col]
        lookup = dict(zip(mapping[id_col], mapping["description"], strict=True))
        df[f"{id_col}_label"] = df[id_col].map(lookup).fillna("Unknown")
    return df


def select_features(df: pd.DataFrame, variance_threshold: float = 0.995) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop near-constant columns and split the rest into numeric and
    categorical features, excluding IDs, raw columns, and sensitive attributes."""
    excluded = set(ID_COLS + DROP_COLS + SENSITIVE_COLS + RAW_ONLY_COLS + [TARGET])
    candidates = [c for c in df.columns if c not in excluded]

    kept: list[str] = []
    dropped: list[str] = []
    for c in candidates:
        top_share = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        (dropped if top_share >= variance_threshold else kept).append(c)
    log.info("Dropped %d near-constant columns: %s", len(dropped), dropped)

    numeric_cols = [c for c in kept if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in kept if c not in numeric_cols]
    for c in categorical_cols:
        df[c] = df[c].astype("category")

    feature_df = df[ID_COLS + SENSITIVE_COLS + numeric_cols + categorical_cols + [TARGET]].copy()
    return feature_df, numeric_cols, categorical_cols


def patient_level_split(
    df: pd.DataFrame, seed: int = SEED, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by patient_nbr so no patient's encounters span two splits."""
    rng = np.random.default_rng(seed)
    patients = df["patient_nbr"].unique()
    rng.shuffle(patients)

    n_train = int(len(patients) * ratios[0])
    n_val = int(len(patients) * ratios[1])
    train_ids = set(patients[:n_train])
    val_ids = set(patients[n_train : n_train + n_val])

    in_train = df["patient_nbr"].isin(train_ids)
    in_val = df["patient_nbr"].isin(val_ids)
    return df[in_train].copy(), df[in_val].copy(), df[~in_train & ~in_val].copy()


def run() -> dict:
    ensure_raw_data()
    raw = load_raw()
    cleaned = clean(raw, load_id_mappings(RAW_DIR / "IDs_mapping.csv"))
    features, numeric_cols, categorical_cols = select_features(cleaned)
    train, val, test = patient_level_split(features)

    splits = {"train": train, "val": val, "test": test}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[a]["patient_nbr"]) & set(splits[b]["patient_nbr"]):
            raise RuntimeError(f"Patient leakage between {a} and {b} splits")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    summary = {
        "raw_rows": len(raw),
        "rows_after_dropping_expired": len(cleaned),
        "unique_patients": int(features["patient_nbr"].nunique()),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "n_features": len(numeric_cols) + len(categorical_cols),
        "sensitive_cols_excluded_from_model": SENSITIVE_COLS,
        "dropped_cols": DROP_COLS,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "target_rate": {
            "overall": float(features[TARGET].mean()),
            **{name: float(part[TARGET].mean()) for name, part in splits.items()},
        },
    }
    with open(PROCESSED_DIR / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote processed splits to %s", PROCESSED_DIR)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    run()
