
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_validation import (  
    FEATURES_9, RANDOM_STATE, SEC4, build_spline_lr, ensure, load_split, log,
    run_env, OUT)

N_REPEATS = 20          # cell 15
N_SPLITS = 5            # cell 15
BOOTSTRAP_N = 1000      # cell 20
BEST_PARAMS = {"clf__C": 1.0, "pre__spline__n_knots": 3}   # from layer1
DELIV = OUT.parent / "deliverables" / "lr_spline"


def find_best_threshold(y_true, y_prob):
    """Verbatim from cell 11 (strategy='youden')."""
    thresholds = np.unique(y_prob)
    best_thr, best_score, best_info = 0.5, -np.inf, {}
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if (precision + sensitivity) > 0 else 0.0)
        youden = sensitivity + specificity - 1
        if youden > best_score:
            best_score, best_thr = youden, thr
            best_info = {"f1": f1, "youden": youden, "precision": precision,
                         "sensitivity": sensitivity, "specificity": specificity}
    return float(best_thr), best_info


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Verbatim from cell 11."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "youden": sensitivity + specificity - 1,
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    d = ensure(DELIV)
    X_tr, X_te, y_tr, y_te, _ = load_split()
    Xtr, Xte = X_tr[FEATURES_9], X_te[FEATURES_9]

    est, _ = build_spline_lr(FEATURES_9)
    est.set_params(**BEST_PARAMS)
    fitted = clone(est).fit(Xtr, y_tr)


    log(f"repeated out-of-fold threshold learning "
        f"({N_REPEATS} repeats x {N_SPLITS} folds) ...")
    t0 = time.time()
    rows, thresholds, youdens = [], [], []
    for rep in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE + rep)
        oof = np.zeros(len(Xtr))
        for tr_idx, va_idx in cv.split(Xtr, y_tr):
            m = clone(fitted).fit(Xtr.iloc[tr_idx], y_tr.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
        thr, info = find_best_threshold(y_tr.values, oof)
        met = compute_metrics(y_tr.values, oof, thr)
        rows.append({"model": "LR_spline", "repeat": rep + 1,
                     "threshold": round(thr, 6),
                     "youden_index": round(info["youden"], 6),
                     **{k: round(met[k], 6) for k in
                        ("auc", "auprc", "accuracy", "balanced_accuracy",
                         "precision", "sensitivity", "specificity", "f1")}})
        thresholds.append(thr)
        youdens.append(info["youden"])

    final_threshold = float(np.median(thresholds))
    final_youden = float(np.median(youdens))
    pd.DataFrame(rows).to_csv(d / "lr_spline_oof_repeats.csv", index=False)
    log(f"  final threshold (median of {N_REPEATS}) = {final_threshold:.6f} "
        f"[{time.time() - t0:.0f}s]")

    # ---- test-set point estimates at that threshold -----------------------
    prob_te = fitted.predict_proba(Xte)[:, 1]
    point = compute_metrics(y_te.values, prob_te, final_threshold)

    # sanity: AUC must match the layer1 run on the same split
    prev = pd.read_csv(OUT / "layer1" / "spline_lr_test_metrics.csv")
    delta = abs(point["auc"] - float(prev["auc"].iloc[0]))
    log(f"  test AUC {point['auc']:.6f} vs layer1 {float(prev['auc'].iloc[0]):.6f} "
        f"(delta {delta:.2e})")
    if delta > 1e-9:
        log("  [WARN] AUC differs from layer1; the fitted model is not identical")

    log(f"bootstrap ({BOOTSTRAP_N}) with threshold fixed ...")
    rng = np.random.RandomState(RANDOM_STATE)
    y_true = y_te.values
    n = len(y_true)
    boot = []
    for b in range(BOOTSTRAP_N):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        met = compute_metrics(y_true[idx], prob_te[idx], final_threshold)
        boot.append({"bootstrap_id": b + 1,
                     **{k: met[k] for k in
                        ("auc", "auprc", "accuracy", "balanced_accuracy",
                         "precision", "sensitivity", "specificity", "f1")}})
    bdf = pd.DataFrame(boot)
    log(f"  valid bootstrap samples: {len(bdf)}")

    ci_rows = []
    for k in ("auc", "auprc", "accuracy", "balanced_accuracy",
              "precision", "sensitivity", "specificity", "f1"):
        lo, hi = np.percentile(bdf[k], [2.5, 97.5])
        ci_rows.append({"model": "LR_spline", "metric": k,
                        "point_estimate": point[k],
                        "ci_lower": float(lo), "ci_upper": float(hi)})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(d / "lr_spline_test_metrics_ci.csv", index=False)
    bdf.to_csv(d / "lr_spline_bootstrap_dist.csv", index=False)

    json.dump({
        "model": "LR_spline",
        "best_params": BEST_PARAMS,
        "final_threshold_median": final_threshold,
        "final_youden_median": final_youden,
        "threshold_repeats_p2_5": float(np.percentile(thresholds, 2.5)),
        "threshold_repeats_p97_5": float(np.percentile(thresholds, 97.5)),
        "n_repeats": N_REPEATS, "n_splits": N_SPLITS,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_rng": "np.random.RandomState(42), re-initialised for this "
                         "model; cell 20 advanced one RandomState across models "
                         "in dict order, which a later model cannot join",
        "confusion_at_threshold": {k: point[k] for k in ("tn", "fp", "fn", "tp")},
        **run_env({"script": "lr_spline_threshold.py"}),
    }, open(d / "lr_spline_threshold.json", "w"), indent=2)

    log("")
    log(f"  threshold          {final_threshold:.6f}")
    for k in ("auc", "auprc", "sensitivity", "specificity", "precision"):
        r = ci[ci.metric == k].iloc[0]
        log(f"  {k:<18} {r.point_estimate:.4f} "
            f"({r.ci_lower:.4f}-{r.ci_upper:.4f})")
    log(f"[OK] -> {d}")


if __name__ == "__main__":
    main()
