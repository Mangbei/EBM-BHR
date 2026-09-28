

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_validation import (  # noqa: E402
    BOOTSTRAP_B, BOOT_SEED, DATA_PATH, ID_COL, MODEL_ORDER, OUT, SEC3, SEC4,
    SEC5, TARGET_COL, all_metrics, ensure, load_split, log, run_env)

DELIV = OUT.parent / "deliverables"
SPLINE = DELIV / "lr_spline"
TABLES = DELIV / "tables"
L0, L1, L2, L3 = (OUT / f"layer{i}" for i in range(4))

# display names used in the manuscript tables
DISPLAY = {"EBM_inter": "EBM_inter", "EBM_main": "EBM_main", "RF": "RF",
           "XGB": "XGBoost", "LGBM": "LightGBM", "LASSO": "LASSO", "LR": "LR",
           "LR_spline": "LR-spline"}
ALL_MODELS = MODEL_ORDER + ["LR_spline"]


def rd(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)


def fmt(point, lo, hi, nd=4):
    return f"{point:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def table1_smd():
    """Table 1: replace the P value column with standardized differences."""
    s = rd(L0 / "table1_smd.csv")
    out = s[["variable", "type", "bhr_positive_mean", "bhr_positive_sd",
             "bhr_negative_mean", "bhr_negative_sd", "smd"]].copy()
    out["smd"] = out["smd"].round(3)
    out.to_csv(TABLES / "table1_smd.csv", index=False)
    return f"Table 1  SMD for {len(out)} variables (|SMD| max {out.smd.abs().max():.3f})"


def table2_performance(spline_ci, spline_meta):
    """Table 2: seven published rows unchanged, plus the spline row."""
    orig = rd(SEC4 / "test_metrics_with_95ci.csv")
    rows = []
    for m in MODEL_ORDER:
        r = orig[orig.model == m].iloc[0]
        rows.append({
            "Model": DISPLAY[m], "Threshold": f"{r.threshold:.4f}",
            "AUC (95% CI)": fmt(r.auc_point, r.auc_ci_low, r.auc_ci_high),
            "AUPRC (95% CI)": fmt(r.auprc_point, r.auprc_ci_low, r.auprc_ci_high),
            "Sensitivity (95% CI)": fmt(r.sensitivity_point, r.sensitivity_ci_low, r.sensitivity_ci_high),
            "Specificity (95% CI)": fmt(r.specificity_point, r.specificity_ci_low, r.specificity_ci_high),
            "Precision (95% CI)": fmt(r.precision_point, r.precision_ci_low, r.precision_ci_high),
            "source": "published main analysis (unchanged)",
        })
    g = spline_ci.set_index("metric")
    rows.append({
        "Model": "LR-spline",
        "Threshold": f"{spline_meta['final_threshold_median']:.4f}",
        **{f"{lab} (95% CI)": fmt(g.loc[k].point_estimate, g.loc[k].ci_lower, g.loc[k].ci_upper)
           for k, lab in [("auc", "AUC"), ("auprc", "AUPRC"),
                          ("sensitivity", "Sensitivity"),
                          ("specificity", "Specificity"),
                          ("precision", "Precision")]},
        "source": "added in revision",
    })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table2_performance.csv", index=False)
    return "Table 2  8 rows (7 published verbatim + LR-spline)"


def paired_calibration_bootstrap():
    """Re-run the layer0 paired bootstrap with the spline model included.

    Indices are drawn before models are evaluated, so the seven original
    models must come out identical; that is asserted, not assumed.
    """
    _, _, _, y_te, _ = load_split()
    yv = y_te.to_numpy()
    pred = rd(SEC4 / "test_pred_prob_all_models.csv")
    spline = pd.read_csv(L1 / "spline_lr_test_pred.csv")
    assert (spline.y_true.to_numpy() == yv).all(), "spline predictions misaligned"

    probs = {m: pred[m].to_numpy() for m in MODEL_ORDER}
    probs["LR_spline"] = spline["prob"].to_numpy()

    metrics = ["brier", "ece", "calibration_intercept", "calibration_slope", "auc"]
    point = {m: all_metrics(yv, probs[m]) for m in ALL_MODELS}

    rng = np.random.default_rng(BOOT_SEED)
    n = len(yv)
    boot = {m: {k: np.empty(BOOTSTRAP_B) for k in metrics} for m in ALL_MODELS}
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)
        yb = yv[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            idx = rng.permutation(n)
            yb = yv[idx]
        for m in ALL_MODELS:
            mm = all_metrics(yb, probs[m][idx])
            for k in metrics:
                boot[m][k][b] = mm[k]
        if (b + 1) % 250 == 0:
            log(f"  paired bootstrap {b + 1}/{BOOTSTRAP_B}")

    rows = []
    for m in ALL_MODELS:
        for k in metrics:
            lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
            rows.append({"model": DISPLAY[m], "metric": k,
                         "point_estimate": point[m][k],
                         "ci_lower": float(lo), "ci_upper": float(hi)})
    ci = pd.DataFrame(rows)

    # reproduction check against the already-published layer0 intervals
    prev = rd(L0 / "calibration_bootstrap_ci.csv")
    bad = []
    for _, r in prev.iterrows():
        now = ci[(ci.model == DISPLAY[r.model]) & (ci.metric == r.metric)]
        if now.empty:
            continue
        now = now.iloc[0]
        for col in ("point_estimate", "ci_lower", "ci_upper"):
            if abs(now[col] - r[col]) > 1e-12:
                bad.append((r.model, r.metric, col, r[col], now[col]))
    if bad:
        log(f"  [FAIL] {len(bad)} interval(s) moved when the spline model was added")
        for x in bad[:5]:
            log(f"    {x}")
        raise SystemExit("paired bootstrap is not reproducible; stopping")
    log("  seven published models reproduce bit-identically")

    pairs = [("EBM_main", "EBM_inter"), ("EBM_main", "LR"), ("EBM_main", "RF"),
             ("EBM_inter", "LR"), ("EBM_inter", "RF"),
             ("EBM_main", "LR_spline"), ("LR_spline", "LR")]
    drows = []
    for a, b_ in pairs:
        for k in metrics:
            dv = boot[a][k] - boot[b_][k]
            lo, hi = np.percentile(dv, [2.5, 97.5])
            drows.append({"comparison": f"{DISPLAY[a]} - {DISPLAY[b_]}", "metric": k,
                          "point_difference": point[a][k] - point[b_][k],
                          "ci_lower": float(lo), "ci_upper": float(hi),
                          "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    diff = pd.DataFrame(drows)

    ci.to_csv(TABLES / "table3_calibration_ci.csv", index=False)
    diff.to_csv(TABLES / "tableS8_paired_differences.csv", index=False)
    return ci, diff


def table4_robustness(spline_holdout_auc):
    """Table 4: holdout AUC beside the repeated-split distribution."""
    raw = pd.read_csv(L2 / "repeated_splits_raw.csv")
    published = {"EBM_inter": 0.8101, "EBM_main": 0.8082, "RF": 0.8059,
                 "XGB": 0.8054, "LGBM": 0.7912, "LASSO": 0.7914, "LR": 0.7914,
                 "LR_spline": spline_holdout_auc}
    rows = []
    for m in ALL_MODELS:
        a = raw[raw.model == m].auc
        rows.append({
            "Model": DISPLAY[m],
            "Internal holdout AUC": round(published[m], 4),
            "Repeated splits: median AUC (2.5-97.5%)":
                f"{np.median(a):.4f} ({np.percentile(a, 2.5):.4f}-{np.percentile(a, 97.5):.4f})",
            "Percentile of holdout value": f"{100 * (a < published[m]).mean():.0f}%",
        })
    df = pd.DataFrame(rows).sort_values("Internal holdout AUC", ascending=False)
    df.to_csv(TABLES / "table4_robustness.csv", index=False)
    return "Table 4  8 models, holdout vs 50 repeated splits"


def tableS1_spline_row():
    """Supplementary Table S1: the spline model's threshold-stability row."""
    rep = pd.read_csv(SPLINE / "lr_spline_oof_repeats.csv")
    row = {"Model": "LR-spline",
           "Mean AUC": round(rep.auc.mean(), 4), "SD of AUC": round(rep.auc.std(ddof=1), 4),
           "Median threshold": round(rep.threshold.median(), 4),
           "Mean threshold": round(rep.threshold.mean(), 4),
           "SD of threshold": round(rep.threshold.std(ddof=1), 4),
           "Mean sensitivity": round(rep.sensitivity.mean(), 4),
           "SD of sensitivity": round(rep.sensitivity.std(ddof=1), 4),
           "Mean specificity": round(rep.specificity.mean(), 4),
           "SD of specificity": round(rep.specificity.std(ddof=1), 4),
           "Mean Youden index": round(rep.youden_index.mean(), 4)}
    pd.DataFrame([row]).to_csv(TABLES / "tableS1_row_lrspline.csv", index=False)
    return f"Table S1  spline row (median threshold {row['Median threshold']:.4f})"


def supplementary_tables():
    out = []
    s3 = pd.read_csv(L2 / "repeated_splits_summary.csv")
    s3["model"] = s3.model.map(DISPLAY).fillna(s3.model)
    s3.to_csv(TABLES / "tableS3_repeated_splits.csv", index=False)
    out.append(f"Table S3  {s3.model.nunique()} models x {s3.metric.nunique()} metrics")

    s4 = pd.read_csv(L3 / "nested_summary.csv")
    s4["model"] = s4.model.map(DISPLAY).fillna(s4.model)
    nested = pd.read_csv(L3 / "nested_raw.csv")
    thr = nested[nested.model == "EBM_main"]["youden_threshold_inner"]
    thr_row = pd.DataFrame([{
        "model": "EBM_main", "metric": "youden_threshold_inner",
        "n_outer_folds": len(thr), "mean": thr.mean(), "median": thr.median(),
        "sd": thr.std(ddof=1), "p2_5": np.percentile(thr, 2.5),
        "p97_5": np.percentile(thr, 97.5)}])
    pd.concat([s4, thr_row], ignore_index=True).to_csv(
        TABLES / "tableS4_nested.csv", index=False)
    pct = 100 * (thr < 0.20275517).mean()
    out.append(f"Table S4  nested metrics + threshold distribution "
               f"(published 0.2028 at {pct:.0f}th percentile)")

    s5 = pd.read_csv(L3 / "nested_feature_selection_summary.csv")
    s5.to_csv(TABLES / "tableS5_selection_stability.csv", index=False)
    out.append(f"Table S5  {int((s5.selection_frequency == 1).sum())} always selected, "
               f"{int((s5.selection_frequency == 0).sum())} never")

    s6 = pd.read_csv(L1 / "all_predictors_test_metrics.csv")
    s6["model"] = s6.model.map(DISPLAY).fillna(s6.model)
    s6.to_csv(TABLES / "tableS6_all14_predictors.csv", index=False)
    out.append(f"Table S6  {len(s6)} models on 14 predictors")

    m = pd.read_csv(L1 / "drop_correlated_metrics.csv")
    i = pd.read_csv(L1 / "drop_correlated_importance.csv")
    m.to_csv(TABLES / "tableS7a_collinearity_metrics.csv", index=False)
    (i[~i.term.str.contains(" x ")]
     .pivot_table(index="term", columns="variant", values="relative_importance")
     .round(4).to_csv(TABLES / "tableS7b_importance_shift.csv"))
    out.append("Table S7  collinearity variants + importance shift")

    s10 = pd.read_csv(L0 / "subgroup_performance.csv")
    s10["model"] = s10.model.map(DISPLAY).fillna(s10.model)
    s10.to_csv(TABLES / "tableS10_subgroups.csv", index=False)
    out.append(f"Table S10  {s10.subgroup.nunique()} subgroups x {s10.model.nunique()} models")

    spec = json.load(open(L0 / "final_model_full_spec.json"))
    json.dump(spec, open(TABLES / "tableS11_full_spec.json", "w"),
              indent=2, ensure_ascii=False)
    flat = []
    for k, v in spec.items():
        flat.append({"section": k,
                     "content": json.dumps(v, ensure_ascii=False)
                     if isinstance(v, (dict, list)) else str(v)})
    pd.DataFrame(flat).to_csv(TABLES / "tableS11_full_spec.csv", index=False)
    out.append("Table S11  full model / scorecard specification")
    return out


def tableS9_scorecard():
    """Scorecard risk groups, operational consequences and net benefit.

    Net benefit is evaluated at each rule's own equivalent probability
    threshold. Evaluating a fixed rule at another rule's threshold is a
    category error and inflates or deflates it arbitrarily.
    """
    sc = rd(SEC5 / "scorecard_test_predictions.csv")
    meta = json.load(open(SEC5 / "scorecard_meta.json", encoding="utf-8-sig"))
    eta, pf = meta["eta_cut"], meta["points_factor"]
    y = sc.y_true.to_numpy()
    score = sc.score_raw.to_numpy()
    n = len(y)
    sig = lambda x: 1 / (1 + np.exp(-x))

    groups = rd(L0 / "riskgroup_wilson_ci.csv")
    groups.to_csv(TABLES / "tableS9a_risk_groups.csv", index=False)

    ops = rd(L0 / "scorecard_operational_metrics.csv")
    ops.to_csv(TABLES / "tableS9b_operational.csv", index=False)

    pred = rd(SEC4 / "test_pred_prob_all_models.csv")

    def nb(mask, pt):
        tp = ((mask) & (y == 1)).sum()
        fp = ((mask) & (y == 0)).sum()
        return tp / n - (fp / n) * (pt / (1 - pt))

    pt_main = sig(eta)
    pt_warn = sig(eta - 5 / pf)
    rules = [
        ("EBM_main", "predicted risk >= 0.2028", pt_main,
         pred.EBM_main.to_numpy() >= 0.20275517),
        ("Scorecard, main boundary", "score >= 0", pt_main, score >= 0),
        ("Scorecard, exploratory boundary", "score >= -5", pt_warn, score >= -5),
    ]
    rows = []
    for name, rule, pt, mask in rules:
        rows.append({"Strategy": name, "Rule": rule,
                     "Threshold probability": round(pt, 6),
                     "Net benefit": round(nb(mask, pt), 4),
                     "Prioritised for BPT": f"{int(mask.sum())}/{n} "
                                            f"({100 * mask.mean():.1f}%)",
                     "Prioritise-all at same threshold":
                         round(nb(np.ones(n, bool), pt), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tableS9c_net_benefit.csv", index=False)
    return (f"Table S9  risk groups + operations + net benefit "
            f"(exploratory boundary threshold {pt_warn:.4f})")


def r2_lln_age_strata():
    """Descriptive answer to reviewer 2's question about the fixed 0.70 ratio.

    Not a GLI-based LLN analysis: those equations are ethnicity-specific
    lookup tables, and a wrong LLN would be worse than none. This shows where
    in the age range misclassification would be most likely, using the cohort's
    own distribution.
    """
    df = pd.read_csv(DATA_PATH)
    ratio = df["FEV1/FVC"]
    bins = [0, 29, 39, 49, 59, 69, 200]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", ">=70"]
    df["age_group"] = pd.cut(df.Age, bins, labels=labels)
    rows = []
    for g, sub in df.groupby("age_group", observed=True):
        near = (sub["FEV1/FVC"] >= 70) & (sub["FEV1/FVC"] < 75)
        rows.append({"Age group": str(g), "n": len(sub),
                     "Median FEV1/FVC (%)": round(sub["FEV1/FVC"].median(), 2),
                     "IQR": f"{sub['FEV1/FVC'].quantile(.25):.2f}-"
                            f"{sub['FEV1/FVC'].quantile(.75):.2f}",
                     "n with ratio 70-75%": int(near.sum()),
                     "% with ratio 70-75%": round(100 * near.mean(), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "r2_lln_age_strata.csv", index=False)
    return (f"R2-3  age strata (cohort minimum ratio {ratio.min():.2f}%, "
            f"n below 70% = {(ratio < 70).sum()})")


def main():
    ensure(TABLES)
    log("building table sources ...")
    notes = []

    spline_ci = pd.read_csv(SPLINE / "lr_spline_test_metrics_ci.csv")
    spline_meta = json.load(open(SPLINE / "lr_spline_threshold.json"))
    spline_auc = float(spline_ci[spline_ci.metric == "auc"].point_estimate.iloc[0])

    notes.append(table1_smd())
    notes.append(table2_performance(spline_ci, spline_meta))
    log("paired calibration bootstrap with the spline model included ...")
    ci, diff = paired_calibration_bootstrap()
    notes.append(f"Table 3  {ci.model.nunique()} models x {ci.metric.nunique()} metrics with CI")
    notes.append(f"Table S8  {len(diff)} paired differences, "
                 f"{int(diff.ci_excludes_zero.sum())} exclude zero")
    notes.append(table4_robustness(spline_auc))
    notes.append(tableS1_spline_row())
    notes.extend(supplementary_tables())
    notes.append(tableS9_scorecard())
    notes.append(r2_lln_age_strata())

    json.dump(run_env({"script": "deliver_tables.py"}),
              open(TABLES / "run_env.json", "w"), indent=2)

    log("")
    for x in notes:
        log(f"  {x}")
    log(f"[OK] {len(list(TABLES.glob('*')))} files -> {TABLES}")


if __name__ == "__main__":
    main()
