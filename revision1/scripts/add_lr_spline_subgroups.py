"""Compute subgroup performance for LR-spline (missing from the original run).

Reproduces the exact setup of revision_validation.py section 0.5:
same 8:2 stratified split (seed 42), same nine subgroup masks,
same bootstrap CI procedure (B=1000, seed 20260809).
Reads the saved LR-spline test predictions (no retraining).
Appends LR_spline rows to a copy of subgroup_performance.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data.csv"
PRED_PATH = REPO / "revision1/results/layer1/spline_lr_test_pred.csv"
ORIG_PATH = REPO / "revision1/results/layer0/subgroup_performance.csv"
OUT_PATH = REPO / "revision1/results/layer0/subgroup_performance_with_spline.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BOOTSTRAP_B = 1000
BOOT_SEED = 20260808
ID_COL, TARGET_COL = "ID", "BHR"


df = pd.read_csv(DATA_PATH)
X = df[[c for c in df.columns if c not in (ID_COL, TARGET_COL)]].copy()
y = df[TARGET_COL].astype(int).copy()
_, X_te, _, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_te = X_te.reset_index(drop=True)
yv = y_te.reset_index(drop=True).to_numpy()


pred = pd.read_csv(PRED_PATH)
assert len(pred) == len(yv), "prediction length does not match test set"
assert (pred["y_true"].to_numpy() == yv).all(), "y_true mismatch vs split"
ps_full = pred["prob"].to_numpy()
print(f"full-test LR_spline AUC = {roc_auc_score(yv, ps_full):.4f} (expect 0.8009)")


med_ratio = float(X_te["FEV1/FVC"].median())
med_fvc = float(X_te["FVC%"].median())
age = X_te["Age"].to_numpy()
subgroups = [
    ("sex: Male=0 == 1", X_te["Male=0"].to_numpy() == 1),
    ("sex: Male=0 == 0", X_te["Male=0"].to_numpy() == 0),
    ("age < 40", age < 40),
    ("age 40-59", (age >= 40) & (age < 60)),
    ("age >= 60", age >= 60),
    (f"FEV1/FVC < median ({med_ratio:.4g})", X_te["FEV1/FVC"].to_numpy() < med_ratio),
    (f"FEV1/FVC >= median ({med_ratio:.4g})", X_te["FEV1/FVC"].to_numpy() >= med_ratio),
    (f"FVC% < median ({med_fvc:.4g})", X_te["FVC%"].to_numpy() < med_fvc),
    (f"FVC% >= median ({med_fvc:.4g})", X_te["FVC%"].to_numpy() >= med_fvc),
]

rng = np.random.default_rng(BOOT_SEED + 1)
rows = []
for label, mask in subgroups:
    ys, ps = yv[mask], ps_full[mask]
    row = {"subgroup": label, "model": "LR_spline", "n": int(mask.sum()),
           "events": int(ys.sum()), "unstable_small_n": bool(ys.sum() < 20),
           "exploratory": True}
    aucs = []
    for _ in range(BOOTSTRAP_B):
        bi = rng.integers(0, len(ys), len(ys))
        if 0 < ys[bi].sum() < len(bi):
            aucs.append(roc_auc_score(ys[bi], ps[bi]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    row.update(auc=float(roc_auc_score(ys, ps)),
               brier=float(brier_score_loss(ys, ps)),
               auc_ci_lower=float(lo), auc_ci_upper=float(hi))
    rows.append(row)
    print(f"{label:28s} n={row['n']:3d} ev={row['events']:3d} "
          f"AUC={row['auc']:.4f} ({lo:.4f}-{hi:.4f}) brier={row['brier']:.4f}")

new = pd.DataFrame(rows)
orig = pd.read_csv(ORIG_PATH)
# sanity: original subgroup sizes match
chk = orig[orig["model"] == "LR"][["subgroup", "n", "events"]].reset_index(drop=True)
assert chk["n"].tolist() == [r["n"] for r in rows], "subgroup sizes differ from original run"
assert chk["events"].tolist() == [r["events"] for r in rows], "event counts differ"

out = pd.concat([orig, new], ignore_index=True)
out.to_csv(OUT_PATH, index=False)
print(f"\nwritten: {OUT_PATH}")
