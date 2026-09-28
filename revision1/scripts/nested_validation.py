
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_validation import (  # noqa: E402  reuse the audited definitions
    CANDIDATES_10, DEMOGRAPHICS_4, DATA_PATH, ID_COL, OUT, TARGET_COL,
    all_metrics, build_models, ensure, log, run_env, tune_fit_eval,
)

TOP_K_IN_FOLD = 5          # cell 7: top_k=5
N_SELECT_REPEATS = 10      # cell 8: n_repeats=10
N_SELECT_SPLITS = 5        # cell 8: n_splits=5
N_RETAIN = 5               # "top 50%" of the ten candidates

PUBLISHED_HOLDOUT_AUC = {
    "LR": 0.7914, "LASSO": 0.7914, "RF": 0.8059, "XGB": 0.8054,
    "LGBM": 0.7912, "EBM_main": 0.8082, "EBM_inter": 0.8101,
}


def select_features_in_fold(X, y, seed):
    """Repeat the published EBM importance screening inside one outer training
    fold. Returns (selected_candidates, hit_counts, mean_importances)."""
    from interpret.glassbox import ExplainableBoostingClassifier

    Xc = X[CANDIDATES_10].reset_index(drop=True)
    yc = y.reset_index(drop=True)
    hits = {f: 0 for f in CANDIDATES_10}
    imps = {f: [] for f in CANDIDATES_10}

    for rep in range(N_SELECT_REPEATS):
        cv = StratifiedKFold(n_splits=N_SELECT_SPLITS, shuffle=True,
                             random_state=seed + rep)
        for tr_idx, _ in cv.split(Xc, yc):
            ebm = ExplainableBoostingClassifier(interactions=0,
                                                random_state=seed + rep)
            ebm.fit(Xc.iloc[tr_idx], yc.iloc[tr_idx])
            names = list(ebm.term_names_)
            vals = list(ebm.term_importances())
            ranked = sorted(zip(names, vals), key=lambda t: -t[1])
            for nm, iv in ranked:
                if nm in imps:
                    imps[nm].append(float(iv))
            for nm, _ in ranked[:TOP_K_IN_FOLD]:
                if nm in hits:
                    hits[nm] += 1

    mean_imp = {f: float(np.mean(v)) if v else 0.0 for f, v in imps.items()}
    order = sorted(CANDIDATES_10, key=lambda f: (-hits[f], -mean_imp[f]))
    return order[:N_RETAIN], hits, mean_imp


def youden_threshold(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10, help="outer repeats")
    a = ap.parse_args()

    d = ensure(OUT / "layer3")
    raw_path = d / "nested_raw.csv"
    sel_path = d / "nested_feature_selection.csv"

    df = pd.read_csv(DATA_PATH)
    X_all = df[[c for c in df.columns if c not in (ID_COL, TARGET_COL)]].copy()
    y_all = df[TARGET_COL].astype(int).copy()

    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["rep"], prev["fold"]))
        log(f"resuming: {len(done)} outer fold(s) already on disk")

    for rep in range(1, a.R + 1):
        seed = 2000 + rep
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, te_idx) in enumerate(outer.split(X_all, y_all), start=1):
            if (rep, fold) in done:
                continue
            t0 = time.time()
            Xtr = X_all.iloc[tr_idx].reset_index(drop=True)
            Xte = X_all.iloc[te_idx].reset_index(drop=True)
            ytr = y_all.iloc[tr_idx].reset_index(drop=True)
            yte = y_all.iloc[te_idx].reset_index(drop=True)

            corr = Xtr[CANDIDATES_10].corr(method="spearman").abs()
            corr_v = corr.to_numpy().copy()
            np.fill_diagonal(corr_v, 0.0)
            top_pair_r = float(corr_v.max())
            i, j = np.unravel_index(np.argmax(corr_v), corr_v.shape)
            top_pair = f"{corr.index[i]}|{corr.columns[j]}"

            t_sel = time.time()
            selected, hits, mean_imp = select_features_in_fold(Xtr, ytr, seed)
            sel_secs = time.time() - t_sel
            feats = DEMOGRAPHICS_4 + [f for f in CANDIDATES_10 if f in selected]

            pd.DataFrame([{
                "rep": rep, "fold": fold, "candidate": f,
                "topk_hits": hits[f],
                "max_hits": N_SELECT_SPLITS * N_SELECT_REPEATS,
                "mean_importance": mean_imp[f],
                "selected": f in selected,
            } for f in CANDIDATES_10]).to_csv(
                sel_path, mode="a", header=not sel_path.exists(), index=False)


            models = build_models(include_spline_for=feats)
            rows = []
            for name, (est, grid) in models.items():
                try:
                    res, prob, best = tune_fit_eval(
                        name, est, grid, Xtr[feats], ytr, Xte[feats], yte,
                        cv_seed=seed)
                    inner_cv = StratifiedKFold(n_splits=5, shuffle=True,
                                               random_state=seed)
                    inner_prob = cross_val_predict(
                        best, Xtr[feats], ytr, cv=inner_cv,
                        method="predict_proba", n_jobs=-1)[:, 1]
                    res["youden_threshold_inner"] = youden_threshold(ytr, inner_prob)
                    res.update(rep=rep, fold=fold, seed=seed,
                               n_selected_features=len(feats),
                               selected_candidates=", ".join(
                                   f for f in CANDIDATES_10 if f in selected),
                               max_abs_spearman=top_pair_r,
                               strongest_pair=top_pair,
                               selection_seconds=round(sel_secs, 1))
                    rows.append(res)
                except Exception as exc:
                    with open(d / "failures.log", "a") as fh:
                        fh.write(f"rep={rep} fold={fold} model={name} err={exc!r}\n")
                    log(f"  [FAIL] rep {rep} fold {fold} {name}: {exc}")
            if rows:
                pd.DataFrame(rows).to_csv(raw_path, mode="a",
                                          header=not raw_path.exists(), index=False)
            ebm_auc = next((r["auc"] for r in rows if r["model"] == "EBM_main"), float("nan"))
            log(f"rep {rep} fold {fold} done in {time.time() - t0:.0f}s "
                f"(selection {sel_secs:.0f}s, EBM_main AUC={ebm_auc:.4f}, "
                f"kept {', '.join(f for f in CANDIDATES_10 if f in selected)})")

    summarise(d, raw_path, sel_path)
    json.dump(run_env({
        "analysis": "fully nested repeated cross-validation",
        "outer": "StratifiedKFold(5, shuffle=True, random_state=2000+rep)",
        "R": a.R,
        "pipeline_repeated_per_fold": [
            "spearman correlation screen (recorded)",
            "EBM importance selection, inner 5-fold x 10 repeats, retain top 5 of 10",
            "GridSearchCV over the published grids, inner 5-fold",
            "Youden threshold from inner cross-validated predictions",
        ],
        "note": "the four demographic variables bypass screening, as in the "
                "published pipeline; the scorecard is NOT rebuilt per fold",
    }), open(d / "run_env.json", "w"), indent=2)
    log(f"[OK] nested validation complete -> {d}")


def summarise(d, raw_path, sel_path):
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path)
    metrics = ["auc", "auprc", "brier", "ece",
               "calibration_intercept", "calibration_slope"]
    rows = []
    for mdl, g in raw.groupby("model"):
        for k in metrics:
            v = g[k].dropna().to_numpy()
            if len(v) == 0:
                continue
            rows.append({"model": mdl, "metric": k, "n_outer_folds": len(v),
                         "mean": float(v.mean()), "median": float(np.median(v)),
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                         "p2_5": float(np.percentile(v, 2.5)),
                         "p97_5": float(np.percentile(v, 97.5))})
    pd.DataFrame(rows).to_csv(d / "nested_summary.csv", index=False)

    comp = []
    for mdl, g in raw.groupby("model"):
        pub = PUBLISHED_HOLDOUT_AUC.get(mdl, np.nan)
        v = g["auc"].dropna().to_numpy()
        comp.append({
            "model": mdl,
            "published_holdout_auc": pub,
            "nested_auc_mean": float(v.mean()),
            "nested_auc_median": float(np.median(v)),
            "nested_auc_p2_5": float(np.percentile(v, 2.5)),
            "nested_auc_p97_5": float(np.percentile(v, 97.5)),
            "optimism_published_minus_nested": (pub - float(v.mean()))
            if pub == pub else np.nan,
        })
    (pd.DataFrame(comp).sort_values("nested_auc_mean", ascending=False)
     .to_csv(d / "nested_vs_published.csv", index=False))

    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        n_folds = sel.groupby("candidate").size().max()
        agg = (sel.groupby("candidate")
               .agg(times_selected=("selected", "sum"),
                    n_folds=("selected", "size"),
                    mean_topk_hits=("topk_hits", "mean"),
                    mean_importance=("mean_importance", "mean"))
               .reset_index())
        agg["selection_frequency"] = agg["times_selected"] / agg["n_folds"]
        (agg.sort_values("selection_frequency", ascending=False)
         .to_csv(d / "nested_feature_selection_summary.csv", index=False))
        log(f"feature-selection stability summarised over {n_folds} outer folds")


if __name__ == "__main__":
    main()
