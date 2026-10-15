"""Readmission models: how much the usual leaks inflate the score, then an
honest model judged on what a hospital needs - calibration, net benefit and
equal performance across patient groups.

Part 1, validation schemes (5-fold, LightGBM, AUC):
  A. rows split at random + total_encounters (leaks the future)
  B. rows split at random, honest features (same patient in train and test)
  C. patients split by group, honest features
  plus logistic regression under scheme C as the transparent baseline.

Part 2, final model: 20% of patients held out. LightGBM fit on the rest, its
probabilities calibrated with isotonic regression on out-of-fold predictions
of the training patients only. Everything below is scored on held-out patients.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CAT, NUM, load

GROUPS = {"race": "race", "age": "age_band", "gender": "gender"}


def gbm(seed=0):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              random_state=seed, verbose=-1)


def logit():
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT)])
    return make_pipeline(pre, LogisticRegression(max_iter=3000, C=0.5))


def cv_auc(d, feats, splitter, groups=None, model=gbm):
    oof = np.zeros(len(d))
    for tr, te in splitter.split(d, d.y, groups):
        m = model().fit(d.iloc[tr][feats], d.y.iloc[tr])
        oof[te] = m.predict_proba(d.iloc[te][feats])[:, 1]
    return roc_auc_score(d.y, oof), oof


def net_benefit(y, p, thresholds):
    """Decision-curve net benefit (Vickers & Elkin 2006) of treating patients with p >= t."""
    n = len(y)
    out = []
    for t in thresholds:
        treat = p >= t
        tp = np.sum(treat & (y == 1)) / n
        fp = np.sum(treat & (y == 0)) / n
        out.append(tp - fp * t / (1 - t))
    return np.array(out)


def group_report(d, y, p, thr):
    rows = {}
    for label, col in GROUPS.items():
        for g, idx in d.groupby(col, observed=True).groups.items():
            pos = d.index.get_indexer(idx)
            yy, pp = y[pos], p[pos]
            if yy.sum() < 30:
                continue
            flagged = pp >= thr
            rows[f"{label}={g}"] = {
                "n": int(len(yy)), "readmit_rate": float(yy.mean()), "mean_predicted": float(pp.mean()),
                "auc": float(roc_auc_score(yy, pp)),
                "sensitivity": float(flagged[yy == 1].mean()), "flag_rate": float(flagged.mean()),
                "ppv": float(yy[flagged].mean()) if flagged.any() else None,
            }
    return rows


def main() -> None:
    d = load()
    honest = NUM + CAT
    leaky = honest + ["total_encounters"]
    res = {"n": int(len(d)), "patients": int(d.patient_nbr.nunique()), "rate": float(d.y.mean())}

    kf, gkf = KFold(5, shuffle=True, random_state=0), GroupKFold(5)
    res["validation"] = {
        "A_random_rows_leaky_feature": cv_auc(d, leaky, kf)[0],
        "B_random_rows_honest": cv_auc(d, honest, kf)[0],
        "C_patient_groups_honest": cv_auc(d, honest, gkf, d.patient_nbr)[0],
        "C_logistic": cv_auc(d, honest, gkf, d.patient_nbr, model=logit)[0],
    }
    print(json.dumps(res["validation"], indent=1), flush=True)

    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(d, d.y, d.patient_nbr))
    tr, te = d.iloc[tr_idx].reset_index(drop=True), d.iloc[te_idx].reset_index(drop=True)
    _, oof = cv_auc(tr, honest, GroupKFold(5), tr.patient_nbr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, tr.y)
    model = gbm().fit(tr[honest], tr.y)
    raw = model.predict_proba(te[honest])[:, 1]
    p = iso.predict(raw)
    y = te.y.to_numpy()
    lr = logit().fit(tr[honest], tr.y).predict_proba(te[honest])[:, 1]

    thresholds = np.round(np.arange(0.05, 0.41, 0.01), 2)
    nb = net_benefit(y, p, thresholds)
    treat_all = y.mean() - (1 - y.mean()) * thresholds / (1 - thresholds)
    thr = 0.20  # flag for follow-up when estimated risk is at least 20%
    flagged = p >= thr
    res["holdout"] = {
        "patients": int(te.patient_nbr.nunique()), "encounters": int(len(te)),
        "auc_lightgbm": roc_auc_score(y, p), "auc_logistic": roc_auc_score(y, lr),
        "brier_raw": brier_score_loss(y, raw), "brier_calibrated": brier_score_loss(y, p),
        "brier_base_rate": brier_score_loss(y, np.full_like(p, tr.y.mean())),
        "threshold": thr, "flag_rate": float(flagged.mean()),
        "sensitivity": float(flagged[y == 1].mean()), "ppv": float(y[flagged].mean()),
        "top_decile_capture": float(y[np.argsort(-p)[: len(y) // 10]].sum() / y.sum()),
        "net_benefit": dict(zip(map(str, thresholds), nb.round(5).tolist())),
        "net_benefit_treat_all": dict(zip(map(str, thresholds), treat_all.round(5).tolist())),
    }
    res["fairness"] = group_report(te, y, p, thr)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=honest).sort_values(ascending=False)
    res["top_features"] = (imp / imp.sum()).round(4).head(10).to_dict()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(res, indent=2))
    np.save("reports/holdout.npy", np.c_[y, p, raw, lr])
    print(json.dumps({k: v for k, v in res["holdout"].items() if not k.startswith("net")}, indent=1))
    for k, v in res["fairness"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print(res["top_features"])


if __name__ == "__main__":
    main()
