from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cliniq.data_prep import clean, group_icd9, load_id_mappings, patient_level_split


@pytest.mark.parametrize(
    ("code", "group"),
    [
        ("250.83", "Diabetes"),
        ("428", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("530", "Digestive"),
        ("996", "Injury"),
        ("715", "Musculoskeletal"),
        ("599", "Genitourinary"),
        ("174", "Neoplasms"),
        ("780", "Other"),
        ("V57", "Other"),
        ("E888", "Other"),
        ("?", "Missing"),
        (np.nan, "Missing"),
    ],
)
def test_group_icd9(code, group):
    assert group_icd9(code) == group


def test_load_id_mappings_handles_comma_only_separators(tmp_path):
    path = tmp_path / "IDs_mapping.csv"
    path.write_text(
        "admission_type_id,description\n1,Emergency\n2,Urgent\n,\n"
        'discharge_disposition_id,description\n1,"Discharged to home"\n11,Expired\n,\n'
        "admission_source_id,description\n7, Emergency Room\n"
    )
    mappings = load_id_mappings(path)
    assert set(mappings) == {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    assert mappings["discharge_disposition_id"].set_index("discharge_disposition_id").loc[1, "description"] == (
        "Discharged to home"
    )
    assert mappings["admission_source_id"]["description"].tolist() == ["Emergency Room"]


def test_clean_drops_expired_and_builds_target_and_labels():
    df = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 11, 1],
            "admission_type_id": [1, 1, 2],
            "admission_source_id": [7, 7, 7],
            "readmitted": ["<30", "NO", ">30"],
            "age": ["[70-80)", "[50-60)", "[10-20)"],
            "diag_1": ["428", "250.1", "?"],
            "diag_2": ["V57", "?", "486"],
            "diag_3": ["?", "?", "?"],
            "medical_specialty": ["Cardiology", "?", np.nan],
        }
    )
    mappings = {
        "admission_type_id": pd.DataFrame({"admission_type_id": [1, 2], "description": ["Emergency", "Urgent"]}),
        "discharge_disposition_id": pd.DataFrame({"discharge_disposition_id": [1], "description": ["Home"]}),
        "admission_source_id": pd.DataFrame({"admission_source_id": [7], "description": ["Emergency Room"]}),
    }
    out = clean(df, mappings)
    assert len(out) == 2  # the expired encounter is gone
    assert out["readmitted_30d"].tolist() == [1, 0]
    assert out["age_midpoint"].tolist() == [75, 15]
    assert out["diag_1_group"].tolist() == ["Circulatory", "Missing"]
    assert out["admission_type_id_label"].tolist() == ["Emergency", "Urgent"]
    assert out["medical_specialty"].tolist() == ["Cardiology", "Unknown"]


def test_patient_level_split_has_no_patient_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_nbr": rng.integers(0, 300, 1_000), "x": np.arange(1_000)})
    train, val, test = patient_level_split(df, seed=1)
    ids = [set(part["patient_nbr"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(df)
