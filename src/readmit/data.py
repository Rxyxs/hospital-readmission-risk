"""Clean the Diabetes 130-US hospitals data and build features.

Target: readmission within 30 days (``readmitted == "<30"``).

Encounters that ended in death or hospice are dropped: those patients cannot
be readmitted, and keeping them would teach the model that the sickest
patients are the safest.

Two patient-history features are built on purpose, one honest and one leaky:
- ``prior_encounters``: the patient's encounters *before* this one (encounter
  ids increase over time), information a hospital has at discharge;
- ``total_encounters``: all of the patient's encounters in the dataset,
  including later ones. A 30-day readmission is itself a later encounter, so
  this feature partly encodes the answer. It is used only to show the leak.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
DATASET = "brandao/diabetes"
EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide",
        "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "insulin",
        "glyburide-metformin"]
CAT = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
       "admission_source_id", "medical_specialty", "payer_code", "A1Cresult", "max_glu_serum",
       "diag_1_group", "diag_2_group", "diag_3_group"] + MEDS + ["change", "diabetesMed"]
NUM = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
       "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
       "prior_encounters", "n_med_changes"]


def download(raw: Path = RAW) -> None:
    if (raw / "diabetic_data.csv").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    subprocess.run([str(kaggle), "datasets", "download", "-d", DATASET, "-p", str(raw), "--unzip"], check=True)


def icd9_group(code) -> str:
    """Map an ICD-9 code to the chapter groups used by Strack et al. (2014)."""
    if pd.isna(code):
        return "missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "other"
    v = float(s)
    if 390 <= v <= 459 or v == 785:
        return "circulatory"
    if 460 <= v <= 519 or v == 786:
        return "respiratory"
    if 520 <= v <= 579 or v == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def load() -> pd.DataFrame:
    download()
    d = pd.read_csv(RAW / "diabetic_data.csv", na_values="?", low_memory=False)
    d = d[~d.discharge_disposition_id.isin(EXPIRED_OR_HOSPICE)].copy()
    d = d[d.gender != "Unknown/Invalid"]
    d["y"] = (d.readmitted == "<30").astype(int)
    d = d.sort_values("encounter_id")
    d["prior_encounters"] = d.groupby("patient_nbr").cumcount()
    d["total_encounters"] = d.groupby("patient_nbr").encounter_id.transform("size")
    for i in (1, 2, 3):
        d[f"diag_{i}_group"] = d[f"diag_{i}"].map(icd9_group)
    d["n_med_changes"] = (d[MEDS].isin(["Up", "Down"])).sum(axis=1)
    d["age_band"] = d.age.str.extract(r"\[(\d+)")[0].astype(int).map(
        lambda a: "<50" if a < 50 else "50-69" if a < 70 else "70+")
    for c in CAT:
        d[c] = d[c].fillna("missing").astype(str).astype("category")
    return d.reset_index(drop=True)


if __name__ == "__main__":
    d = load()
    print(f"{len(d):,} encounters, {d.patient_nbr.nunique():,} patients, readmitted <30d {d.y.mean():.3%}")
    print("leak check, readmission rate by total encounters:",
          d.groupby(d.total_encounters.clip(upper=4)).y.mean().round(3).to_dict())
    print("honest, by prior encounters:", d.groupby(d.prior_encounters.clip(upper=3)).y.mean().round(3).to_dict())
