#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        
import numpy as np                        
import pandas as pd                       
from sklearn.calibration import calibration_curve   
from sklearn.metrics import (average_precision_score, confusion_matrix,  
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_validation import (  
    MODEL_ORDER, OUT, SEC4, ensure, load_split, log, run_env)

DELIV = OUT.parent / "deliverables"
FIGS = DELIV / "figures"
DISPLAY = {"EBM_inter": "EBM_inter", "EBM_main": "EBM_main", "RF": "RF",
           "XGB": "XGB", "LGBM": "LGBM", "LASSO": "LASSO", "LR": "LR",
           "LR_spline": "LR-spline"}
DCA_THRESHOLDS = np.arange(0.01, 0.25, 0.01)   # cell 23


def load_predictions():
    _, _, _, y_te, _ = load_split()
    yv = y_te.to_numpy()
    pred = pd.read_csv(SEC4 / "test_pred_prob_all_models.csv", encoding="utf-8-sig")
    assert (pred.y_true.to_numpy() == yv).all(), "saved predictions misaligned"
    spline = pd.read_csv(OUT / "layer1" / "spline_lr_test_pred.csv")
    assert (spline.y_true.to_numpy() == yv).all(), "spline predictions misaligned"
    probs = {m: pred[m].to_numpy() for m in MODEL_ORDER}
    probs["LR_spline"] = spline["prob"].to_numpy()
    return yv, probs


def fig_roc(y, probs):
    plt.figure(figsize=(7, 6), dpi=300)
    for m, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{DISPLAY[m]} (AUC={roc_auc_score(y, p):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Set")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(FIGS / "test_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    return "test_roc_curves.png  8 curves"


def fig_calibration(y, probs):
    plt.figure(figsize=(7, 6), dpi=300)
    for m, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=DISPLAY[m])
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curves on Test Set")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGS / "test_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    return "test_calibration_curves.png  8 curves, 10 quantile bins"


def fig_dca(y, probs):
    """Reference strategies are renamed to match the clinical action here:
    the decision is testing priority, not treatment."""
    n = len(y)
    rows = []
    for m, p in probs.items():
        for pt in DCA_THRESHOLDS:
            pred = (p >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            rows.append({"model": DISPLAY[m], "threshold": pt,
                         "net_benefit": (tp / n) - (fp / n) * (pt / (1 - pt))})
    prevalence = y.mean()
    for pt in DCA_THRESHOLDS:
        rows.append({"model": "Prioritize all", "threshold": pt,
                     "net_benefit": prevalence - (1 - prevalence) * (pt / (1 - pt))})
        rows.append({"model": "Prioritize none", "threshold": pt,
                     "net_benefit": 0.0})
    dca = pd.DataFrame(rows)
    dca.to_csv(DELIV / "tables" / "dca_long_df_with_spline.csv", index=False)

    plt.figure(figsize=(8, 6), dpi=300)
    for m in dca.model.unique():
        sub = dca[dca.model == m]
        plt.plot(sub.threshold, sub.net_benefit, label=m)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis on Test Set")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(FIGS / "dca_curves_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    return (f"dca_curves_test.png  8 curves + prioritise-all/none, "
            f"thresholds {DCA_THRESHOLDS[0]:.2f}-{DCA_THRESHOLDS[-1]:.2f}")


def verify_against_published(y, probs):
    """The redrawn curves must carry the published numbers, not new ones."""
    published = {"EBM_inter": 0.8101, "EBM_main": 0.8082, "RF": 0.8059,
                 "XGB": 0.8054, "LGBM": 0.7912, "LASSO": 0.7914, "LR": 0.7914}
    bad = []
    for m, want in published.items():
        got = roc_auc_score(y, probs[m])
        if abs(got - want) > 5e-5:
            bad.append((m, want, got))
    if bad:
        raise SystemExit(f"redrawn curves do not match published AUCs: {bad}")
    log("  published AUCs reproduce from the saved predictions (tol 5e-5)")


def main():
    ensure(FIGS)
    ensure(DELIV / "tables")
    y, probs = load_predictions()
    log(f"loaded {len(probs)} models, n={len(y)}, positives={int(y.sum())}")
    verify_against_published(y, probs)
    for note in (fig_roc(y, probs), fig_calibration(y, probs), fig_dca(y, probs)):
        log(f"  {note}")
    json.dump(run_env({
        "script": "deliver_figures.py",
        "sources": ["ebm_output/section4/test_pred_prob_all_models.csv",
                    "revision1/results/layer1/spline_lr_test_pred.csv"],
        "note": "no model was refitted; curves are drawn from saved predictions",
    }), open(FIGS / "run_env.json", "w"), indent=2)
    log(f"[OK] -> {FIGS}")


if __name__ == "__main__":
    main()
