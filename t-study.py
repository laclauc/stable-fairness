"""
t-study: Effect of the target prediction rate t on SOCP fairness.

Sweeps t in (0,1) for DP and EO on Adult. For each t (including t_ERM),
reports accuracy, unfairness gap, and deployment gap.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results";  RES_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import SOCPFairnessMixture, ERMBaseline
from datasets.loaders import load_adult
from shared.evaluation import compute_targets, eval_stats, test_stats
from shared.plotting import save_fig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH     = ROOT / "datasets" / "adult" / "adult_norm.csv"
N_SEEDS       = 20
T_GRID        = np.linspace(0.05, 0.95, 25)
PLOT_FROM_CSV = False

SOCP_DP_PARAMS = dict(K=30, pi_bar=0.02, m=60, delta=1e-5, C=1.0, constraint="DP")
SOCP_EO_PARAMS = dict(K=30, pi_bar=0.15, m=60, delta=1e-5, C=1.0, constraint="EO")

CSV_PATH = RES_DIR / "t_study_adult.csv"

# ---------------------------------------------------------------------------
# Sweep  (skipped when PLOT_FROM_CSV=True)
# ---------------------------------------------------------------------------

if not PLOT_FROM_CSV:
    X_all, G_all, Y_all = load_adult(DATA_PATH)
    print(f"Adult: n={len(Y_all)}, P(G=1)={G_all.mean():.3f}, P(Y=1)={Y_all.mean():.3f}")

    rows = []
    t_total = time.time()

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed+1}/{N_SEEDS} ===", flush=True)
        X_tr, X_te, G_tr, G_te, y_tr, y_te = train_test_split(
            X_all, G_all, Y_all, test_size=0.3, random_state=seed
        )
        p1, p11, p01 = test_stats(G_te, y_te)

        erm = ERMBaseline(C=1.0)
        erm.fit(X_tr, G_tr, y_tr)
        erm_p = erm.predict_proba(X_te)
        t_dp_erm, t_eo_erm = compute_targets(erm, X_tr, G_tr, y_tr)
        print(f"  ERM t_dp={t_dp_erm:.3f}  t_eo={t_eo_erm:.3f}", flush=True)

        for c, t_erm, params in [
            ("DP", t_dp_erm, SOCP_DP_PARAMS),
            ("EO", t_eo_erm, SOCP_EO_PARAMS),
        ]:
            erm_acc, erm_u, erm_dep = eval_stats(erm_p, G_te, y_te, t_erm, p1, p11, p01, c)
            rows.append(dict(model="ERM", constraint=c, seed=seed,
                             t=t_erm, is_tERM=True,
                             acc=erm_acc, u=erm_u, deploy=erm_dep))
            print(f"  ERM {c}  acc={erm_acc:.3f}  u={erm_u:.3f}  dep={erm_dep:.3f}", flush=True)

            t_list = sorted(set(round(float(t), 8) for t in T_GRID) | {round(t_erm, 8)})
            for t in t_list:
                t0 = time.time()
                try:
                    socp = SOCPFairnessMixture(**params, t=t)
                    socp.fit(X_tr, G_tr, y_tr, seed=seed)
                    p    = socp.predict_proba(X_te)
                    acc, u, dep = eval_stats(p, G_te, y_te, t, p1, p11, p01, c)
                except RuntimeError:
                    acc, u, dep = np.nan, np.nan, np.nan
                    print(f"    {c} t={t:.3f} INFEASIBLE", flush=True)
                rows.append(dict(model="SOCP", constraint=c, seed=seed,
                                 t=t, is_tERM=(t == round(t_erm, 8)),
                                 acc=acc, u=u, deploy=dep))
                print(f"    {c} t={t:.3f}  acc={acc:.3f}  u={u:.3f}  dep={dep:.3f}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"Results saved to {CSV_PATH}")

# ---------------------------------------------------------------------------
# Load from CSV
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH.name}.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SOCP_COLOR = "#1f77b4"
ERM_COLOR  = "#d62728"
MS         = 9

for c in ("DP", "EO"):
    socp_df = df[(df.model == "SOCP") & (df.constraint == c)]
    erm_df  = df[(df.model == "ERM")  & (df.constraint == c)]

    t_erm_m  = float(erm_df["t"].mean())
    t_erm_s  = float(erm_df["t"].std())
    grid_t   = np.sort(socp_df[~socp_df["is_tERM"]]["t"].unique())
    all_t    = np.sort(np.append(grid_t, t_erm_m))   # one point for t_ERM

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    handles = []

    for ax, metric, ylabel in [
        (axes[0], "acc",    "Accuracy"),
        (axes[1], "u",      "Unfairness"),
        (axes[2], "deploy", "Deployment"),
    ]:
        means, stds = [], []
        for t in all_t:
            if np.isclose(t, t_erm_m):
                # aggregate all seeds' t_ERM results regardless of exact t
                vals = socp_df[socp_df["is_tERM"]][metric].values
            else:
                vals = socp_df[(~socp_df["is_tERM"]) & np.isclose(socp_df["t"], t)][metric].values
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))

        means = np.array(means)
        stds  = np.array(stds)

        # Dashed connecting line + shaded std band
        line, = ax.plot(all_t, means, color=SOCP_COLOR, linestyle="--", linewidth=1.5, zorder=1)
        ax.fill_between(all_t, means - stds, means + stds, color=SOCP_COLOR, alpha=0.12, zorder=0)
        # Markers on top
        ax.plot(all_t, means, "*", color=SOCP_COLOR, markersize=MS, zorder=3)

        # ERM horizontal reference + std band
        erm_val = float(erm_df[metric].mean())
        erm_std = float(erm_df[metric].std())
        hl = ax.axhline(erm_val, color=ERM_COLOR, lw=2, ls="--", label="ERM")
        ax.fill_between([all_t[0] - 0.02, all_t[-1] + 0.02],
                        erm_val - erm_std, erm_val + erm_std,
                        color=ERM_COLOR, alpha=0.12, zorder=0)

        # Vertical dotted line + std band + value annotation at mean t_ERM
        ax.axvspan(t_erm_m - t_erm_s, t_erm_m + t_erm_s, color="black", alpha=0.08, zorder=0)
        ax.axvline(t_erm_m, color="black", lw=1.5, ls=":", zorder=1)
        ax.annotate(f"$t_{{\\rm ERM}}={t_erm_m:.2f}$",
                    xy=(t_erm_m, 0), xycoords=("data", "axes fraction"),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, va="bottom", rotation=90)

        ax.set_xlabel(r"$t$", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(all_t[0] - 0.02, all_t[-1] + 0.02)
        ax.grid(True, alpha=0.2)

        if not handles:
            from matplotlib.lines import Line2D
            socp_handle = Line2D([0], [0], color=SOCP_COLOR, linestyle="--", linewidth=1.5,
                                 marker="*", markersize=MS)
            handles = [socp_handle, hl]

    fig.legend(handles, ["STABLE", "ERM"],
               loc="upper center", ncol=2,
               fontsize=12, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.08))

    save_fig(fig, f"t_study_adult_{c}", FIG_DIR)

print("Figures saved.")
