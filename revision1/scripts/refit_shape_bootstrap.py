
import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "revision1" / "results" / "shape_refit"
FIGDIR = REPO / "revision1" / "deliverables" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)

N_BOOT, SEED, SUPPORT_Q = 1000, 42, (2.5, 97.5)
PANEL_ORDER = ["Age", "BMI", "EOS", "FEV1/FVC", "FVC%", "Height",
               "MEF50%", "MMEF%", "Male=0"]
BINARY = {"Male=0"}
DISPLAY = {"Male=0": "Sex"}

d = pd.read_csv(REPO / "data.csv")
model = joblib.load(REPO / "ebm_output/section3/saved_final_models/EBM_main.joblib")
FEATS = list(model.feature_names_in_)
Xtr, _, ytr, _ = train_test_split(d[FEATS], d["BHR"], test_size=0.2,
                                  stratify=d["BHR"], random_state=SEED)
Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)


def term_index(m, feat):
    fi = list(m.feature_names_in_).index(feat)
    for ti, f in enumerate(m.term_features_):
        if list(f) == [fi]:
            return ti, fi
    raise KeyError(feat)


def bin_scores(m, feat):
    """Per-bin scores and cut points, with the sentinel bins excluded."""
    ti, fi = term_index(m, feat)
    e = m.explain_global()
    names = list(e.data()["names"])
    scores = np.asarray(e.data(names.index(feat))["scores"], dtype=float).ravel()
    b = m.bins_[fi]
    cuts = np.asarray(b[0] if isinstance(b, (list, tuple)) else b, dtype=float).ravel()
    return scores, cuts


def evaluate(scores, cuts, x):
    return scores[np.clip(np.digitize(x, cuts, right=True), 0, len(scores) - 1)]


grid, ref = {}, {}
for f in FEATS:
    s, c = bin_scores(model, f)
    if f in BINARY:
        xs = np.array([0.0, 1.0])
    else:
        edges = np.concatenate([[Xtr[f].min()], c, [Xtr[f].max()]])
        xs = (edges[:-1] + edges[1:]) / 2
    grid[f] = xs
    ref[f] = evaluate(s, c, xs)

rng = np.random.default_rng(SEED)
y = ytr.values.astype(int)
i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
boot = {f: [] for f in FEATS}

for b in range(N_BOOT):
    idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
    rng.shuffle(idx)
    mb = clone(model).fit(Xtr.iloc[idx].reset_index(drop=True),
                          ytr.iloc[idx].reset_index(drop=True))
    for f in FEATS:
        s, c = bin_scores(mb, f)
        boot[f].append(evaluate(s, c, grid[f]))
    if (b + 1) % 100 == 0:
        print(f"[{b+1}/{N_BOOT}]", flush=True)

rows = []
stat = {}
for f in FEATS:
    B = np.vstack(boot[f])
    mean, lo, hi = B.mean(0), np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
    stat[f] = (mean, lo, hi)
    for x, r, m_, l_, h_ in zip(grid[f], ref[f], mean, lo, hi):
        rows.append(dict(feature=f, x=x, score_final_log_odds=r,
                         boot_mean=m_, ci_lo=l_, ci_hi=h_, n_boot_used=N_BOOT))
pd.DataFrame(rows).to_csv(OUTDIR / "ebm_main_shape_table_corrected.csv", index=False)


BLUE, ORANGE = "#1f77b4", "#F5C08A"
fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
for ax, f in zip(axes.ravel(), PANEL_ORDER):
    xs, r = grid[f], ref[f]
    mean, lo, hi = stat[f]
    ax.axhline(0, color="grey", lw=.7, ls=":", zorder=1)
    if f in BINARY:
        # 0 codes male in this dataset; the axis is labelled in words
        n = [int((Xtr[f] == v).sum()) for v in (0, 1)]
        ax.errorbar([0, 1], mean, yerr=[mean - lo, hi - mean], fmt="o",
                    color=BLUE, ecolor="#9ecae1", elinewidth=3, capsize=0,
                    ms=9, zorder=3, label="Bootstrap mean, 95% CI")
        for xv, rv in zip([0, 1], r):
            ax.hlines(rv, xv - .12, xv + .12, color="#444", lw=2, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Male\n(n={n[0]})", f"Female\n(n={n[1]})"])
        ax.set_xlim(-.5, 1.5); ax.set_xlabel("Sex")
        ax.set_title("Sex (binary)", fontsize=9)
    else:
        xt = Xtr[f].values
        q_lo, q_hi = np.percentile(xt, SUPPORT_Q)
        ax2 = ax.twinx()
        ax2.hist(xt, bins=30, color=ORANGE, alpha=.55, zorder=0)
        ax2.set_ylabel("Train distribution (count)", fontsize=7, color="#B07A3A")
        ax2.tick_params(labelsize=6, colors="#B07A3A")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.fill_between(xs, lo, hi, color="#9ecae1", alpha=.5, zorder=2)
        ax.plot(xs, mean, color=BLUE, lw=1.8, zorder=3, label="Bootstrap mean, 95% CI")
        ax.plot(xs, r, color="#444", lw=1.4, ls="--", zorder=4, label="Ref fit (full train)")
        for a, b_ in ((xs.min(), q_lo), (q_hi, xs.max())):
            if b_ > a:
                ax.axvspan(a, b_, color="0.85", alpha=.55, zorder=1)
        ax.set_xlabel(f); ax.set_title(f, fontsize=9)
    ax.set_ylabel("Log-odds contribution", fontsize=8)
    ax.tick_params(labelsize=7)

h, l = axes.ravel()[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, .04, 1, 1])
fig.savefig(FIGDIR / "ebm_main_shapes_grid.png", dpi=300)
print("done")
