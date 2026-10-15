"""README figures: leak comparison, calibration, decision curve, group audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/img")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads(Path("reports/results.json").read_text())
    h = np.load("reports/holdout.npy")
    y, p = h[:, 0], h[:, 1]

    # 1. Leak
    v = r["validation"]
    labels = ["random rows +\nall-time visit count", "random rows,\nhonest features",
              "split by patient,\nhonest features", "split by patient,\nlogistic"]
    vals = [v["A_random_rows_leaky_feature"], v["B_random_rows_honest"],
            v["C_patient_groups_honest"], v["C_logistic"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(labels, vals, color=["#E15759", "#BAB0AC", "#4E79A7", "#A0CBE8"])
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center")
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("AUC, 5-fold")
    ax.set_title("A feature that counts future visits inflates AUC by 0.12")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "leak.png", dpi=130)
    plt.close(fig)

    # 2. Calibration (deciles of predicted risk)
    q = np.quantile(p, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    mp = [p[b == i].mean() for i in range(10)]
    obs = [y[b == i].mean() for i in range(10)]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 0.45], [0, 0.45], "k:", lw=1)
    ax.plot(mp, obs, "o-", color="#4E79A7")
    ax.set_xlabel("predicted risk (decile mean)")
    ax.set_ylabel("observed 30-day readmission")
    ax.set_title("Calibration on held-out patients")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=130)
    plt.close(fig)

    # 3. Decision curve
    nb = r["holdout"]["net_benefit"]
    ta = r["holdout"]["net_benefit_treat_all"]
    t = np.array([float(k) for k in nb])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(t * 100, [nb[k] * 100 for k in nb], color="#4E79A7", lw=2, label="model")
    ax.plot(t * 100, [ta[k] * 100 for k in ta], color="#F28E2B", lw=1.5, label="follow up everyone")
    ax.axhline(0, color="k", lw=1, label="follow up no one")
    ax.set_ylim(-2, 7)
    ax.set_xlabel("risk threshold for follow-up (%)")
    ax.set_ylabel("net benefit (readmissions caught\nper 100 patients, net of false alarms)")
    ax.set_title("Decision curve on held-out patients")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "decision_curve.png", dpi=130)
    plt.close(fig)

    # 4. Group audit
    f = {k: v for k, v in r["fairness"].items()}
    names = list(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    yy = np.arange(len(names))
    axes[0].barh(yy - 0.2, [f[n]["readmit_rate"] * 100 for n in names], 0.4, label="observed", color="#BAB0AC")
    axes[0].barh(yy + 0.2, [f[n]["mean_predicted"] * 100 for n in names], 0.4, label="predicted", color="#4E79A7")
    axes[0].set_xlabel("30-day readmission (%)")
    axes[0].set_title("Calibration by group")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[1].barh(yy, [f[n]["sensitivity"] * 100 for n in names], color="#59A14F")
    for i, n in enumerate(names):
        axes[1].text(f[n]["sensitivity"] * 100 + 0.5, i, f"n={f[n]['n']:,}", va="center", fontsize=8)
    axes[1].set_xlabel("readmissions flagged at 20% threshold (%)")
    axes[1].set_title("Sensitivity by group")
    axes[0].set_yticks(yy, names)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "groups.png", dpi=130)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
