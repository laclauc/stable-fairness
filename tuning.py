"""
Hyperparameter grid search (delta x pi_bar) for any dataset.

Usage:
    python tuning.py --dataset adult     --constraint DP
    python tuning.py --dataset compas    --constraint EO
    python tuning.py --dataset acs       --constraint DP
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

from utils import SOCPFairnessMixture, ERMBaseline
from shared.evaluation import eg_predict_proba, compute_targets, eval_stats, test_stats
from shared.plotting import plot_heatmap, save_fig

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    required=True,
                    choices=["adult","compas","german","communities","acs"])
parser.add_argument("--constraint", required=True, choices=["DP","EO"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tune-km", action="store_true",
                    help="Also grid-search over K and m (slower)")
args = parser.parse_args()

DATASET    = args.dataset
CONSTRAINT = args.constraint
SEED       = args.seed
TUNE_KM    = args.tune_km

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

from datasets.loaders import load_adult, load_compas, load_german, load_communities

DATA_ROOTS = {
    "adult":       ROOT / "datasets" / "adult"       / "adult_norm.csv",
    "compas":      ROOT / "datasets" / "compas"      / "Processed_Compas.csv",
}

if DATASET == "acs":
    from folktables import ACSDataSource, ACSIncome
    from sklearn.model_selection import train_test_split as tts
    src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    raw = src.get_data(states=["CA"], download=True)
    X_all, y_all, G_raw = ACSIncome.df_to_numpy(raw)
    y_all = y_all.astype(int); G_all = (G_raw == 1).astype(int)
    _, idx = tts(np.arange(len(y_all)), test_size=30_000/len(y_all), random_state=SEED,
                 stratify=np.stack([y_all, G_all], axis=1))
    X_all, G_all, y_all = X_all[idx], G_all[idx], y_all[idx]
else:
    loaders = {"adult": load_adult, "compas": load_compas,
               "german": load_german, "communities": load_communities}
    X_all, G_all, y_all = loaders[DATASET](DATA_ROOTS[DATASET])

print(f"Dataset: {DATASET}, n={len(y_all)}, P(G=1)={G_all.mean():.3f}, P(Y=1)={y_all.mean():.3f}")

stratify = y_all if DATASET in ("compas", "adult") else None
X_tr, X_te, G_tr, G_te, y_tr, y_te = train_test_split(
    X_all, G_all, y_all, test_size=0.3, random_state=SEED, stratify=stratify
)
p1, p11, p01 = test_stats(G_te, y_te)

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

erm = ERMBaseline(C=1.0, task="classification")
erm.fit(X_tr, G_tr, y_tr)
t_dp, t_eo = compute_targets(erm, X_tr, G_tr, y_tr)
t_target   = t_dp if CONSTRAINT == "DP" else t_eo

eg = ExponentiatedGradient(
    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    constraints=(DemographicParity() if CONSTRAINT=="DP" else EqualizedOdds()),
    eps=0.01,
)
eg.fit(X_tr, y_tr, sensitive_features=G_tr)

acc_erm, u_erm, dep_erm = eval_stats(erm.predict_proba(X_te), G_te, y_te, t_target, p1, p11, p01, CONSTRAINT)
acc_eg,  u_eg,  dep_eg  = eval_stats(eg_predict_proba(eg, X_te), G_te, y_te, t_target, p1, p11, p01, CONSTRAINT)
print(f"ERM: acc={acc_erm:.3f} U={u_erm:.3f} deploy={dep_erm:.3f}")
print(f"EG:  acc={acc_eg:.3f}  U={u_eg:.3f}  deploy={dep_eg:.3f}")

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

DELTAS  = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
PI_BARS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

K_VALS = [10, 20, 30, 50]   if TUNE_KM else [30]
M_VALS = [30, 60, 90, 120]  if TUNE_KM else [60]

# Shape: (pi_bar, delta) or (pi_bar, delta, K, m) flattened to 4-D internally,
# but we store results keyed by (K, m) pairs for easy reporting.
from itertools import product as iproduct

grid_u      = {}
grid_acc    = {}
grid_deploy = {}

total = len(PI_BARS) * len(DELTAS) * len(K_VALS) * len(M_VALS)
print(f"\nRunning {total} fits ({CONSTRAINT}"
      + (", K×m sweep" if TUNE_KM else "") + ")...")

for K, m in iproduct(K_VALS, M_VALS):
    key = (K, m)
    grid_u[key]      = np.full((len(PI_BARS), len(DELTAS)), np.nan)
    grid_acc[key]    = np.full_like(grid_u[key], np.nan)
    grid_deploy[key] = np.full_like(grid_u[key], np.nan)

    for i, pi_bar in enumerate(PI_BARS):
        for j, delta in enumerate(DELTAS):
            try:
                socp = SOCPFairnessMixture(K=K, pi_bar=pi_bar, m=m, delta=delta,
                                           t=t_target, C=1.0, constraint=CONSTRAINT)
                socp.fit(X_tr, G_tr, y_tr, seed=SEED)
                acc, u, dep = eval_stats(socp.predict_proba(X_te), G_te, y_te,
                                         t_target, p1, p11, p01, CONSTRAINT)
                grid_u[key][i,j] = u; grid_acc[key][i,j] = acc; grid_deploy[key][i,j] = dep
                print(f"  K={K:3d} m={m:4d} pi_bar={pi_bar:.2f} delta={delta:.0e}  "
                      f"acc={acc:.3f} U={u:.3f} deploy={dep:.3f}")
            except RuntimeError:
                print(f"  K={K:3d} m={m:4d} pi_bar={pi_bar:.2f} delta={delta:.0e}  INFEASIBLE")

# ---------------------------------------------------------------------------
# Heatmaps  (one pair of panels per (K, m) combination)
# ---------------------------------------------------------------------------

delta_labels = [f"{d:.0e}" for d in DELTAS]
pibar_labels = [str(p) for p in PI_BARS]

all_u   = np.concatenate([g.ravel() for g in grid_u.values()])
all_acc = np.concatenate([g.ravel() for g in grid_acc.values()])
u_max   = max(u_erm,   float(np.nanmax(all_u)))   if not np.all(np.isnan(all_u))   else u_erm
acc_min = min(acc_erm, float(np.nanmin(all_acc)))  if not np.all(np.isnan(all_acc)) else acc_erm

for (K, m) in iproduct(K_VALS, M_VALS):
    key = (K, m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_heatmap(axes[0], grid_u[key],   "Blues_r", 0.0,     u_max,   delta_labels, pibar_labels)
    plot_heatmap(axes[1], grid_acc[key], "Greens",  acc_min, acc_erm, delta_labels, pibar_labels)
    km_tag = f"K{K}_m{m}" if TUNE_KM else ""
    axes[0].set_title(fr"$U(h)$ — {CONSTRAINT} $\downarrow$" + (f"  (K={K}, m={m})" if TUNE_KM else ""), fontsize=12)
    axes[1].set_title(r"Accuracy $\uparrow$" + (f"  (K={K}, m={m})" if TUNE_KM else ""), fontsize=12)
    save_fig(fig, f"{DATASET}_heatmap_{CONSTRAINT}" + (f"_{km_tag}" if TUNE_KM else ""), FIG_DIR)

# ---------------------------------------------------------------------------
# Best configs
# ---------------------------------------------------------------------------

print("\n--- Best configurations ---")

def _best(grid_dict, metric="min"):
    best_val = np.inf if metric == "min" else -np.inf
    best_cfg = None
    for (K, m), grid in grid_dict.items():
        if np.all(np.isnan(grid)):
            continue
        idx = np.nanargmin(grid) if metric == "min" else np.nanargmax(grid)
        i, j = np.unravel_index(idx, grid.shape)
        val = grid[i, j]
        if (metric == "min" and val < best_val) or (metric == "max" and val > best_val):
            best_val, best_cfg = val, (K, m, i, j)
    return best_val, best_cfg

best_u_val, best_u_cfg = _best(grid_u, "min")
if best_u_cfg:
    K, m, i, j = best_u_cfg
    print(f"Best U(h):   K={K} m={m} pi_bar={PI_BARS[i]} delta={DELTAS[j]:.0e} "
          f"-> U={best_u_val:.3f} acc={grid_acc[(K,m)][i,j]:.3f} deploy={grid_deploy[(K,m)][i,j]:.3f}")

best_dep_val, best_dep_cfg = _best(grid_deploy, "min")
if best_dep_cfg:
    K, m, i, j = best_dep_cfg
    print(f"Best deploy: K={K} m={m} pi_bar={PI_BARS[i]} delta={DELTAS[j]:.0e} "
          f"-> deploy={best_dep_val:.3f} U={grid_u[(K,m)][i,j]:.3f} acc={grid_acc[(K,m)][i,j]:.3f}")