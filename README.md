[English ] | [[Español](README.es.md) ]

# 30-day hospital readmission: a risk model a hospital could actually use

![CI](https://github.com/Rxyxs/hospital-readmission-risk/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

99,340 real inpatient encounters of diabetic patients from 130 US hospitals (1999-2008),
with demographics, diagnoses, lab tests, medications and prior visits. 11.4% are readmitted
within 30 days. Hospitals want to know at discharge who needs extra follow-up.

A readmission model is easy to make look good and hard to make useful. This project is about
the second part: removing the leaks that inflate the score, then judging the model the way a
clinical team would, by calibration, net benefit and equal performance across patient groups.

## Results at a glance

| | |
|---|---|
| AUC with the common leaks | 0.805 |
| **AUC, honest, on held-out patients** | **0.671** (logistic regression 0.654) |
| Patients flagged at 20% estimated risk | 7.8%, of whom 28% are readmitted (2.4x the base rate) |
| Readmissions in the riskiest 10% | 23% |
| Net benefit vs following up everyone | positive at every threshold from 5% to 39% |
| Weakest groups | patients 70+ (AUC 0.64) and patients with no recorded race (3% of their readmissions flagged) |

## 1. Two leaks, one of them large

![Leak](docs/img/leak.png)

**Counting a patient's visits across the whole dataset leaks the answer.** A 30-day readmission
is itself another visit, so "number of encounters for this patient" partly encodes the target:
patients seen once have a 4.3% readmission rate *by construction*. Counting only visits before
the current one (9.0% for first visits) is what a hospital knows at discharge. The leaky feature
lifts AUC from 0.684 to 0.805.

**Splitting rows instead of patients barely matters here.** 16,773 patients have more than one
encounter, so a random split puts the same person in training and test, but AUC moves only
from 0.684 to 0.685. Both splits are reported; all results below hold out whole patients.

Two cleaning decisions also follow from the question: encounters that ended in death or hospice
(2,423) are removed, since those patients cannot be readmitted, and diagnoses are grouped into
ICD-9 chapters as in Strack et al. (2014).

## 2. Calibrated, and useful at realistic thresholds

20% of patients (13,998 people, 19,739 encounters) are held out entirely. LightGBM is fit on the
rest; isotonic recalibration on out-of-fold predictions turned out unnecessary (Brier 0.0939 raw,
0.0940 recalibrated, against 0.0983 for predicting the base rate to everyone).

![Calibration](docs/img/calibration.png)

![Decision curve](docs/img/decision_curve.png)

A decision curve (Vickers & Elkin, 2006) answers whether acting on the model beats the simple
policies, without inventing costs: the threshold itself encodes how many false alarms one
caught readmission is worth. At a 10% threshold the model yields a net benefit of 3.1 readmissions
caught per 100 patients against 1.2 for following up everyone; above 11%, following up everyone
is worse than doing nothing while the model stays positive up to 39%.

The most informative inputs are prior inpatient visits, discharge destination, admitting
specialty, prior encounters and the primary diagnosis.

## 3. Does it work equally for everyone?

![Groups](docs/img/groups.png)

At a 20% flagging threshold, on held-out patients:

- **Calibration holds for the large groups.** Predicted and observed rates match within 0.3 points for men,
  women, patients 50+, African American and Caucasian patients, and within 0.8 points for under-50s.
- **Race: no gap between the two main groups.** 19% of readmissions are flagged for African
  American patients and 20% for Caucasian patients.
- **Age: older patients are served worse.** AUC 0.64 for 70+ against 0.72 under 50, and 18% of
  readmissions flagged against 28%. The patients with the highest readmission rate are the ones
  the model separates least well.
- **Missing race is a real blind spot.** For the 445 encounters with no recorded race the model
  over-predicts (8.9% predicted, 7.0% observed) and flags 3% of their readmissions. Missing
  demographic data tends to mark patients whose records are incomplete in other ways too.
- Hispanic patients (n = 400) show a high sensitivity (30%) but are over-predicted as well; with
  33 readmissions in the group, the numbers are noisy.

## Method

```
diabetic_data.csv ──► drop death/hospice · ICD-9 chapters · prior visits only (data.py)
     │
     ├── leak study: random rows + all-time visit count | random rows | patient groups (5-fold)
     │
     ├── hold out 20% of patients
     │      LightGBM ── isotonic check on out-of-fold scores of training patients
     │      logistic regression baseline
     │
     └── held-out evaluation: AUC, Brier, calibration, decision curve,
                              per-group calibration, AUC, sensitivity and PPV
```

## Data

Diabetes 130-US hospitals for years 1999-2008, Strack et al. (2014),
"Impact of HbA1c Measurement on Hospital Readmission Rates", *BioMed Research International*;
originally published on the UCI Machine Learning Repository. Kaggle mirror
[brandao/diabetes](https://www.kaggle.com/datasets/brandao/diabetes), license CC0.
Downloaded October 2026; not redistributed here.

## Reproduce

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m readmit.data      # downloads with ~/.kaggle/kaggle.json
python -m readmit.model     # ~3 min
python -m readmit.figures
pytest -q tests
```

## Limitations

- The data ends in 2008 and coding practices have changed (ICD-10 since 2015 in the US). The
  workflow transfers; the fitted model would not.
- Readmissions to other hospitals are not recorded, so the target undercounts.
- A net benefit says the model beats the simple policies at a given trade-off; whether follow-up
  actually prevents readmissions is a separate, causal question this data cannot answer.

## License

Code: MIT. Data: CC0 (Kaggle mirror of the UCI dataset), not redistributed.
