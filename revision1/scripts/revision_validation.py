
import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
ID_COL = "ID"
TARGET_COL = "BHR"

REPO = Path(__file__).resolve().parents[2]
DATA_PATH = REPO / "data.csv"
SEC3 = REPO / "ebm_output" / "section3"
SEC4 = REPO / "ebm_output" / "section4"
SEC5 = REPO / "ebm_output" / "section5" / "scorecard"
OUT = REPO / "revision1" / "results"


FEATURES_9 = ["Male=0", "Age", "BMI", "Height", "EOS",
              "FVC%", "FEV1/FVC", "MMEF%", "MEF50%"]


CANDIDATES_10 = ["EOS", "FVC%", "FEV1%", "FEV1/FVC", "PEF%",
                 "MMEF%", "MEF75%", "MEF50%", "MEF25%", "WBC"]
DEMOGRAPHICS_4 = ["Male=0", "Age", "BMI", "Height"]

FEATURES_14 = DEMOGRAPHICS_4 + CANDIDATES_10

MODEL_ORDER = ["LR", "LASSO", "RF", "XGB", "LGBM", "EBM_main", "EBM_inter"]

BOOTSTRAP_B = 1000
BOOT_SEED = 20260808


PUBLISHED = {
    "EBM_inter": dict(auc=0.8101, brier=0.1165, ece=0.0257, slope=0.9891, intercept=-0.0407),
    "EBM_main":  dict(auc=0.8082, brier=0.1169, ece=0.0233, slope=1.0318, intercept=-0.0045),
    "RF":        dict(auc=0.8059, brier=0.1180, ece=0.0311, slope=1.0138, intercept=-0.0304),
    "XGB":       dict(auc=0.8054, brier=0.1176, ece=0.0317, slope=0.9058, intercept=-0.1412),
    "LGBM":      dict(auc=0.7912, brier=0.1205, ece=0.0392, slope=0.7802, intercept=-0.2541),
    "LASSO":     dict(auc=0.7914, brier=0.1209, ece=0.0462, slope=0.9240, intercept=-0.1081),
    "LR":        dict(auc=0.7914, brier=0.1210, ece=0.0487, slope=0.9207, intercept=-0.1114),
}
TOL = 1e-4

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split():
    """Reproduce the notebook's train/test split exactly (cell 1)."""
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return (X_tr.reset_index(drop=True), X_te.reset_index(drop=True),
            y_tr.reset_index(drop=True), y_te.reset_index(drop=True), df.shape[0])


def ece_quantile(y_true, y_prob, n_bins=10):
    """Verbatim re-implementation of the notebook's ECE (cell 22):
    10 equal-frequency bins built by slicing the probability-sorted vector."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    order = np.argsort(y_prob)
    p_s, y_s = y_prob[order], y_true[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    ece = 0.0
    for b in range(n_bins):
        s, e = edges[b], edges[b + 1]
        if e <= s:
            continue
        ece += (e - s) / n * abs(y_s[s:e].mean() - p_s[s:e].mean())
    return float(ece)


def calibration_intercept_slope(y_true, y_prob):
    """Unpenalised logistic recalibration on logit(p) (cell 22)."""
    p = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True, penalty=None,
                            solver="lbfgs", max_iter=1000)
    lr.fit(logit_p, np.asarray(y_true))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def all_metrics(y_true, y_prob):
    icpt, slope = calibration_intercept_slope(y_true, y_prob)
    return dict(
        auc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=ece_quantile(y_true, y_prob),
        calibration_intercept=icpt,
        calibration_slope=slope,
    )


def wilson_ci(k, n, alpha=0.05):
    from scipy.stats import norm
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((c - h) / d, (c + h) / d)


def run_env(extra=None):
    import sklearn
    import scipy
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "random_state": RANDOM_STATE,
        "bootstrap_seed": BOOT_SEED,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for mod in ("interpret", "xgboost", "lightgbm"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "unavailable"
    if extra:
        info.update(extra)
    return info


N_JOBS = max(1, (os.cpu_count() or 2) - 1)


EST_N_JOBS = 1


def build_models(include_spline_for=None):
    """Return {name: (estimator, param_grid)}. Grids are NOT modified."""
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    def logreg_pipe(penalty, solver, max_iter):
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(penalty=penalty, solver=solver,
                                                    max_iter=max_iter,
                                                    random_state=RANDOM_STATE))])

    m = {}
    m["LR"] = (logreg_pipe("l2", "lbfgs", 4000),
               {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None]})
    m["LASSO"] = (logreg_pipe("l1", "saga", 8000),
                  {"clf__C": [0.05, 0.1, 0.5, 1.0], "clf__class_weight": [None]})
    m["RF"] = (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=EST_N_JOBS),
               {"n_estimators": [300, 600], "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 3], "max_features": ["sqrt"],
                "class_weight": [None, "balanced"]})
    m["XGB"] = (XGBClassifier(random_state=RANDOM_STATE, n_jobs=EST_N_JOBS,
                              eval_metric="logloss", tree_method="hist",
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
                {"n_estimators": [300, 600], "learning_rate": [0.03, 0.1],
                 "max_depth": [3, 4], "min_child_weight": [1, 5]})
    m["LGBM"] = (LGBMClassifier(random_state=RANDOM_STATE, n_jobs=EST_N_JOBS, verbose=-1,
                                subsample=0.8, colsample_bytree=0.8),
                 {"n_estimators": [300, 600], "learning_rate": [0.03, 0.1],
                  "num_leaves": [15, 31], "min_child_samples": [10, 30],
                  "class_weight": [None, "balanced"]})
    m["EBM_main"] = (ExplainableBoostingClassifier(random_state=RANDOM_STATE, interactions=0),
                     {"max_bins": [64, 128], "min_samples_leaf": [1, 5]})
    m["EBM_inter"] = (ExplainableBoostingClassifier(random_state=RANDOM_STATE, interactions=10),
                      {"max_bins": [64, 128], "min_samples_leaf": [1, 5]})

    if include_spline_for is not None:
        m["LR_spline"] = build_spline_lr(include_spline_for)
    return m


def build_spline_lr(feature_list):
    """Restricted-cubic-spline logistic regression: an interpretable nonlinear
    benchmark for R1 major comment 4. Binary columns are passed through."""
    binary = [f for f in feature_list if f == "Male=0"]
    continuous = [f for f in feature_list if f not in binary]
    pre = ColumnTransformer([
        ("spline", SplineTransformer(degree=3, include_bias=False,
                                     extrapolation="linear"), continuous),
        ("bin", "passthrough", binary),
    ])
    pipe = Pipeline([("pre", pre), ("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                                max_iter=8000,
                                                random_state=RANDOM_STATE))])
    grid = {"pre__spline__n_knots": [3, 4, 5], "clf__C": [0.1, 1.0, 10.0]}
    return pipe, grid


def tune_fit_eval(name, est, grid, X_tr, y_tr, X_te, y_te, cv_seed=RANDOM_STATE):
    """One GridSearchCV (5-fold, refit on AUC) then evaluate on the held-out set.
    Identical scheme to notebook cell 14."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    gs = GridSearchCV(clone(est), grid,
                      scoring={"auc": "roc_auc", "auprc": "average_precision"},
                      refit="auc", cv=cv, n_jobs=-1, return_train_score=False)
    gs.fit(X_tr, y_tr)
    prob = gs.best_estimator_.predict_proba(X_te)[:, 1]
    res = all_metrics(y_te, prob)
    res["model"] = name
    res["best_params"] = json.dumps(gs.best_params_, sort_keys=True, default=str)
    res["cv_auc"] = float(gs.cv_results_["mean_test_auc"][gs.best_index_])
    return res, prob, gs.best_estimator_

def stage0():
    d = ensure(OUT / "layer0")
    X_tr, X_te, y_tr, y_te, n_total = load_split()


    pred = pd.read_csv(SEC4 / "test_pred_prob_all_models.csv", encoding="utf-8-sig")
    ok_rows = len(pred) == len(y_te)
    ok_y = ok_rows and bool((pred["y_true"].to_numpy() == y_te.to_numpy()).all())
    sc = pd.read_csv(SEC5 / "scorecard_test_predictions.csv", encoding="utf-8-sig")
    ok_sc = len(sc) == len(y_te) and bool((sc["y_true"].to_numpy() == y_te.to_numpy()).all())
    check = {
        "n_total": int(n_total),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "test_positives": int(y_te.sum()),
        "train_prevalence": float(y_tr.mean()), "test_prevalence": float(y_te.mean()),
        "pred_file_rows_match": ok_rows,
        "pred_file_y_aligned": ok_y,
        "scorecard_file_y_aligned": ok_sc,
    }
    if not (ok_y and ok_sc):
        json.dump(check, open(d / "alignment_check.json", "w"), indent=2)
        sys.exit("[FAIL] saved predictions are not row-aligned with the reproduced "
                 "split; every downstream task is invalid. Stopping.")
    log(f"alignment OK  (test n={len(y_te)}, positives={int(y_te.sum())})")


    yv = y_te.to_numpy()
    rows, deviations = [], []
    for mdl in MODEL_ORDER:
        met = all_metrics(yv, pred[mdl].to_numpy())
        met["model"] = mdl
        rows.append(met)
        for k, pk in [("auc", "auc"), ("brier", "brier"), ("ece", "ece"),
                      ("calibration_slope", "slope"), ("calibration_intercept", "intercept")]:
            diff = abs(met[k] - PUBLISHED[mdl][pk])
            if diff > TOL:
                deviations.append({"model": mdl, "metric": k, "recomputed": met[k],
                                   "published": PUBLISHED[mdl][pk], "abs_diff": diff})
    pd.DataFrame(rows).to_csv(d / "recomputed_test_metrics.csv", index=False)
    check["published_metrics_reproduced"] = len(deviations) == 0
    check["deviations"] = deviations
    json.dump(check, open(d / "alignment_check.json", "w"), indent=2)
    if deviations:
        log(f"[WARN] {len(deviations)} metric(s) differ from the published values "
            f"by more than {TOL}; see alignment_check.json")
    else:
        log("published test metrics reproduced exactly (tol 1e-4)")


    log(f"paired bootstrap, B={BOOTSTRAP_B} ...")
    rng = np.random.default_rng(BOOT_SEED)
    n = len(yv)
    metrics_of_interest = ["brier", "ece", "calibration_intercept", "calibration_slope", "auc"]
    boot = {m: {k: np.empty(BOOTSTRAP_B) for k in metrics_of_interest} for m in MODEL_ORDER}
    probs = {m: pred[m].to_numpy() for m in MODEL_ORDER}
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)            # one index set shared by all models -> paired
        yb = yv[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            idx = rng.permutation(n)
            yb = yv[idx]
        for m in MODEL_ORDER:
            mm = all_metrics(yb, probs[m][idx])
            for k in metrics_of_interest:
                boot[m][k][b] = mm[k]
        if (b + 1) % 200 == 0:
            log(f"  bootstrap {b + 1}/{BOOTSTRAP_B}")

    ci_rows = []
    point = {r["model"]: r for r in rows}
    for m in MODEL_ORDER:
        for k in metrics_of_interest:
            lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
            ci_rows.append({"model": m, "metric": k, "point_estimate": point[m][k],
                            "ci_lower": float(lo), "ci_upper": float(hi),
                            "bootstrap_B": BOOTSTRAP_B})
    pd.DataFrame(ci_rows).to_csv(d / "calibration_bootstrap_ci.csv", index=False)

    pairs = [("EBM_main", "EBM_inter"), ("EBM_main", "LR"), ("EBM_main", "RF"),
             ("EBM_inter", "LR"), ("EBM_inter", "RF")]
    diff_rows = []
    for a, b_ in pairs:
        for k in metrics_of_interest:
            dvec = boot[a][k] - boot[b_][k]
            lo, hi = np.percentile(dvec, [2.5, 97.5])
            diff_rows.append({
                "comparison": f"{a} - {b_}", "metric": k,
                "point_difference": point[a][k] - point[b_][k],
                "ci_lower": float(lo), "ci_upper": float(hi),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "bootstrap_B": BOOTSTRAP_B,
            })
    pd.DataFrame(diff_rows).to_csv(d / "calibration_paired_differences.csv", index=False)
    log("calibration bootstrap written")


    score = sc["score_raw"].to_numpy()
    ysc = sc["y_true"].to_numpy()
    groups = [("low_risk (score < -5)", score < -5),
              ("intermediate (-5 <= score < 0)", (score >= -5) & (score < 0)),
              ("scorecard positive (score >= 0)", score >= 0)]
    grp_rows = []
    for label, mask in groups:
        k, nn = int(ysc[mask].sum()), int(mask.sum())
        lo, hi = wilson_ci(k, nn)
        grp_rows.append({"risk_group": label, "n": nn, "events": k,
                         "event_rate": k / nn, "wilson_ci_lower": lo,
                         "wilson_ci_upper": hi})
    pd.DataFrame(grp_rows).to_csv(d / "riskgroup_wilson_ci.csv", index=False)

    total_n, total_pos = len(ysc), int(ysc.sum())
    strat_rows = []
    for label, mask in [("test only if score >= 0", score >= 0),
                        ("test only if score >= -5", score >= -5),
                        ("test everyone (current practice)", np.ones_like(score, dtype=bool))]:
        tested = int(mask.sum())
        found = int(ysc[mask].sum())
        missed = total_pos - found
        untested = total_n - tested
        untested_events = missed
        strat_rows.append({
            "strategy": label,
            "tests_per_100_patients": 100 * tested / total_n,
            "tests_avoided_per_100_patients": 100 * untested / total_n,
            "positives_detected_per_100_patients": 100 * found / total_n,
            "positives_missed_per_100_patients": 100 * missed / total_n,
            "share_of_all_positives_missed": missed / total_pos,
            "ppv_among_tested": found / tested if tested else np.nan,
            "npv_among_not_tested": (1 - untested_events / untested) if untested else np.nan,
            "number_needed_to_test_per_positive": tested / found if found else np.nan,
        })
    pd.DataFrame(strat_rows).to_csv(d / "scorecard_operational_metrics.csv", index=False)
    log("scorecard operational metrics written")


    cov = X_te.reset_index(drop=True)
    med_ratio = float(cov["FEV1/FVC"].median())
    med_fvc = float(cov["FVC%"].median())
    age = cov["Age"].to_numpy()
    subgroups = [
        ("sex: Male=0 == 1", cov["Male=0"].to_numpy() == 1),
        ("sex: Male=0 == 0", cov["Male=0"].to_numpy() == 0),
        ("age < 40", age < 40),
        ("age 40-59", (age >= 40) & (age < 60)),
        ("age >= 60", age >= 60),
        (f"FEV1/FVC < median ({med_ratio:.4g})", cov["FEV1/FVC"].to_numpy() < med_ratio),
        (f"FEV1/FVC >= median ({med_ratio:.4g})", cov["FEV1/FVC"].to_numpy() >= med_ratio),
        (f"FVC% < median ({med_fvc:.4g})", cov["FVC%"].to_numpy() < med_fvc),
        (f"FVC% >= median ({med_fvc:.4g})", cov["FVC%"].to_numpy() >= med_fvc),
    ]
    rng = np.random.default_rng(BOOT_SEED + 1)
    sg_rows = []
    for label, mask in subgroups:
        ys = yv[mask]
        for m in MODEL_ORDER:
            ps = probs[m][mask]
            row = {"subgroup": label, "model": m, "n": int(mask.sum()),
                   "events": int(ys.sum()),
                   "unstable_small_n": bool(ys.sum() < 20),
                   "exploratory": True}
            if 0 < ys.sum() < len(ys):
                row["auc"] = float(roc_auc_score(ys, ps))
                row["brier"] = float(brier_score_loss(ys, ps))
                aucs = []
                for _ in range(BOOTSTRAP_B):
                    bi = rng.integers(0, len(ys), len(ys))
                    if 0 < ys[bi].sum() < len(bi):
                        aucs.append(roc_auc_score(ys[bi], ps[bi]))
                if aucs:
                    lo, hi = np.percentile(aucs, [2.5, 97.5])
                    row["auc_ci_lower"], row["auc_ci_upper"] = float(lo), float(hi)
            sg_rows.append(row)
    pd.DataFrame(sg_rows).to_csv(d / "subgroup_performance.csv", index=False)
    log("subgroup performance written")


    df = pd.read_csv(DATA_PATH)
    y_all = df[TARGET_COL].astype(int)
    smd_rows = []
    for col in [c for c in df.columns if c not in (ID_COL, TARGET_COL)]:
        a, b_ = df.loc[y_all == 1, col], df.loc[y_all == 0, col]
        binary = set(pd.unique(df[col].dropna())) <= {0, 1}
        if binary:
            p1, p0 = a.mean(), b_.mean()
            denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
            smd = (p1 - p0) / denom if denom > 0 else np.nan
        else:
            denom = np.sqrt((a.var(ddof=1) + b_.var(ddof=1)) / 2)
            smd = (a.mean() - b_.mean()) / denom if denom > 0 else np.nan
        smd_rows.append({
            "variable": col, "type": "binary" if binary else "continuous",
            "bhr_positive_n": int(len(a)), "bhr_negative_n": int(len(b_)),
            "bhr_positive_mean": float(a.mean()), "bhr_negative_mean": float(b_.mean()),
            "bhr_positive_sd": float(a.std(ddof=1)), "bhr_negative_sd": float(b_.std(ddof=1)),
            "smd": float(smd), "abs_smd": float(abs(smd)),
        })
    (pd.DataFrame(smd_rows).sort_values("abs_smd", ascending=False)
     .to_csv(d / "table1_smd.csv", index=False))
    log("Table 1 SMDs written")

    spec = {
        "data_split": {"n_total": int(n_total), "n_train": int(len(y_tr)),
                       "n_test": int(len(y_te)), "test_size": TEST_SIZE,
                       "stratified_by": TARGET_COL, "random_state": RANDOM_STATE,
                       "validation_type": "randomly held-out internal test set "
                                          "(same centre, same period, same population)"},
        "primary_feature_set": FEATURES_9,
        "ebm_screening_candidates": CANDIDATES_10,
        "ebm_screening_rule": {
            "model": "ExplainableBoostingClassifier(interactions=0)",
            "scheme": "5-fold cross-validation repeated 10 times on the training set "
                      "(random_state = 42 + repeat index)",
            "top_k": 5,
            "description": "for every one of the 50 fold-fits, the five variables with "
                           "the highest EBM global importance were recorded; the "
                           "retention rule was the number of top-5 hits out of 50",
            "note": "the four demographic variables were retained on clinical grounds "
                    "and never entered the EBM screening step",
        },
        "hyperparameter_search": {
            "scheme": "single 5-fold StratifiedKFold(shuffle=True, random_state=42) "
                      "GridSearchCV on the full training set, refit on AUC",
            "note": "feature selection and hyperparameter tuning were performed once "
                    "on the full training set; the subsequent 20x5 repeated "
                    "out-of-fold analysis was used only for threshold stability and "
                    "is therefore conditional on that fixed configuration",
            "class_imbalance": "no oversampling or undersampling; class_weight='balanced' "
                               "was a grid candidate for RF and LGBM but was not selected",
            "selected": json.load(open(SEC3 / "best_params.json", encoding="utf-8-sig")),
        },
        "ece_definition": {"n_bins": 10, "strategy": "equal-frequency (quantile)",
                           "formula": "sum_b (n_b/n) * |mean(y_b) - mean(p_b)|"},
        "calibration_definition": "unpenalised logistic regression of the outcome on "
                                  "logit(clip(p, 1e-6, 1-1e-6)); intercept and slope",
        "scorecard_meta": json.load(open(SEC5 / "scorecard_meta.json", encoding="utf-8-sig")),
        "scorecard_bin_spec": json.load(open(SEC5 / "scorecard_bin_spec.json", encoding="utf-8-sig")),
        "environment_of_this_validation_run": run_env(),
    }
    json.dump(spec, open(d / "final_model_full_spec.json", "w"), indent=2, ensure_ascii=False)
    log("model specification written")

    json.dump(run_env({"stage": 0}), open(d / "run_env.json", "w"), indent=2)
    log(f"[OK] stage 0 complete -> {d}")



def stage1():
    d = ensure(OUT / "layer1")
    X_tr, X_te, y_tr, y_te, _ = load_split()

    # ---- 1.1 spline logistic regression on the nine primary predictors ----
    log("spline logistic regression (nine predictors) ...")
    est, grid = build_spline_lr(FEATURES_9)
    res, prob, _ = tune_fit_eval("LR_spline", est, grid,
                                 X_tr[FEATURES_9], y_tr, X_te[FEATURES_9], y_te)
    pd.DataFrame([res]).to_csv(d / "spline_lr_test_metrics.csv", index=False)
    pd.DataFrame({"y_true": y_te.to_numpy(), "prob": prob}).to_csv(
        d / "spline_lr_test_pred.csv", index=False)
    log(f"  LR_spline AUC={res['auc']:.4f} brier={res['brier']:.4f} "
        f"slope={res['calibration_slope']:.4f} params={res['best_params']}")

    # ---- 1.2 all 14 prespecified predictors, no EBM-based screening -------
    log("all-14-predictor sensitivity analysis, 8 models ...")
    models = build_models(include_spline_for=FEATURES_14)
    rows14 = []
    for name, (est, grid) in models.items():
        t0 = time.time()
        r, _, _ = tune_fit_eval(name, est, grid,
                                X_tr[FEATURES_14], y_tr, X_te[FEATURES_14], y_te)
        r["feature_set"] = "all_14_prespecified"
        r["seconds"] = round(time.time() - t0, 1)
        rows14.append(r)
        log(f"  {name:10s} AUC={r['auc']:.4f} ({r['seconds']}s)")
    df14 = pd.DataFrame(rows14).sort_values("auc", ascending=False)
    df14.to_csv(d / "all_predictors_test_metrics.csv", index=False)

    # ---- 1.3 EBM variants dropping one of the correlated spirometry pair --
    log("correlated-predictor sensitivity analysis ...")
    from interpret.glassbox import ExplainableBoostingClassifier
    drop_rows, imp_rows = [], []
    variants = {
        "EBM_main_9_features": FEATURES_9,
        "EBM_main_drop_MEF50pct": [f for f in FEATURES_9 if f != "MEF50%"],
        "EBM_main_drop_MMEFpct": [f for f in FEATURES_9 if f != "MMEF%"],
    }
    for label, feats in variants.items():
        est = ExplainableBoostingClassifier(random_state=RANDOM_STATE, interactions=0)
        grid = {"max_bins": [64, 128], "min_samples_leaf": [1, 5]}
        r, _, best = tune_fit_eval(label, est, grid,
                                   X_tr[feats], y_tr, X_te[feats], y_te)
        r["n_features"] = len(feats)
        r["features"] = ", ".join(feats)
        drop_rows.append(r)
        imps = best.term_importances()
        names = best.term_names_
        total = float(np.sum(imps)) or 1.0
        for nm, iv in sorted(zip(names, imps), key=lambda t: -t[1]):
            imp_rows.append({"variant": label, "term": nm,
                             "importance": float(iv),
                             "relative_importance": float(iv) / total})
        log(f"  {label:26s} AUC={r['auc']:.4f} brier={r['brier']:.4f}")
    pd.DataFrame(drop_rows).to_csv(d / "drop_correlated_metrics.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(d / "drop_correlated_importance.csv", index=False)

    json.dump(run_env({"stage": 1, "features_14": FEATURES_14,
                       "features_9": FEATURES_9}),
              open(d / "run_env.json", "w"), indent=2)
    log(f"[OK] stage 1 complete -> {d}")


def stage2(R):
    d = ensure(OUT / "layer2")
    raw_path = d / "repeated_splits_raw.csv"
    fail_path = d / "failures.log"

    df = pd.read_csv(DATA_PATH)
    X_all = df[[c for c in df.columns if c not in (ID_COL, TARGET_COL)]].copy()
    y_all = df[TARGET_COL].astype(int).copy()

    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(prev["replicate"].unique())
        log(f"resuming: {len(done)} replicate(s) already on disk")

    for r in range(1, R + 1):
        if r in done:
            continue
        seed = 1000 + r
        t0 = time.time()
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=seed, stratify=y_all)
        Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
        ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)

        models = build_models(include_spline_for=FEATURES_9)
        rows = []
        for name, (est, grid) in models.items():
            try:
                res, _, _ = tune_fit_eval(name, est, grid,
                                          Xtr[FEATURES_9], ytr,
                                          Xte[FEATURES_9], yte, cv_seed=seed)
                res.update(replicate=r, split_seed=seed)
                rows.append(res)
            except Exception as exc:                     # never abort the whole run
                with open(fail_path, "a") as fh:
                    fh.write(f"replicate={r} model={name} error={exc!r}\n")
                log(f"  [FAIL] replicate {r} model {name}: {exc}")
        if rows:
            out = pd.DataFrame(rows)
            out.to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
        log(f"replicate {r}/{R} done in {time.time() - t0:.0f}s "
            f"(EBM_main AUC="
            f"{next((x['auc'] for x in rows if x['model'] == 'EBM_main'), float('nan')):.4f})")

    # ---- summary ----------------------------------------------------------
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        metrics = ["auc", "auprc", "brier", "ece",
                   "calibration_intercept", "calibration_slope"]
        summ = []
        for mdl, g in raw.groupby("model"):
            for k in metrics:
                v = g[k].dropna().to_numpy()
                if len(v) == 0:
                    continue
                summ.append({"model": mdl, "metric": k, "n_replicates": len(v),
                             "mean": float(v.mean()), "median": float(np.median(v)),
                             "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                             "p2_5": float(np.percentile(v, 2.5)),
                             "p97_5": float(np.percentile(v, 97.5)),
                             "min": float(v.min()), "max": float(v.max())})
        (pd.DataFrame(summ).sort_values(["metric", "model"])
         .to_csv(d / "repeated_splits_summary.csv", index=False))


        freq = (raw.groupby(["model", "best_params"]).size()
                .reset_index(name="times_selected")
                .sort_values(["model", "times_selected"], ascending=[True, False]))
        freq["share"] = freq["times_selected"] / freq.groupby("model")["times_selected"].transform("sum")
        freq.to_csv(d / "hyperparameter_selection_frequency.csv", index=False)

        # per-replicate model ranking by AUC
        rank = raw.copy()
        rank["auc_rank"] = rank.groupby("replicate")["auc"].rank(ascending=False)
        (rank.groupby("model")["auc_rank"]
         .agg(mean_rank="mean", median_rank="median",
              times_best=lambda s: int((s == 1).sum()))
         .reset_index().sort_values("mean_rank")
         .to_csv(d / "repeated_splits_model_ranking.csv", index=False))

    json.dump(run_env({"stage": 2, "R": R, "seed_rule": "1000 + replicate index",
                       "feature_set": FEATURES_9,
                       "note": "the nine-predictor set is held fixed; hyperparameters "
                               "are re-tuned inside every replicate. Feature selection "
                               "is NOT repeated per replicate -- that question is "
                               "addressed separately by the stage-1 all-predictor "
                               "analysis."}),
              open(d / "run_env.json", "w"), indent=2)
    log(f"[OK] stage 2 complete -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["0", "1", "2", "all"])
    ap.add_argument("--R", type=int, default=50,
                    help="number of repeated splits in stage 2")
    a = ap.parse_args()
    ensure(OUT)
    if a.stage in ("0", "all"):
        stage0()
    if a.stage in ("1", "all"):
        stage1()
    if a.stage in ("2", "all"):
        stage2(a.R)


if __name__ == "__main__":
    main()
