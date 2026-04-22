"""
Synthetic data experiment — phi curves and Pareto frontiers.
N_SEEDS controls variance bands. Set N_SEEDS=1 for a quick single run.

Config: edit the constants at the top.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

from utils import SOCPFairnessMixture, ERMBaseline, phi_dp
from shared.evaluation import eg_predict_proba
from shared.plotting import save_fig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SEEDS  = 10
N_TRAIN  = 4000
N_TEST   = 8000
PI_TRAIN = 0.2
T_TARGET = 0.5
PI_GRID  = np.linspace(0.0, 1.0, 200)

SOCP_PARAMS    = dict(K=30, pi_bar=0.05, m=60, delta=1e-3, t=T_TARGET, C=1.0)
DELTA_GRID     = np.logspace(-5, -1, 25)   # for Pareto sweep
PI_BAR_PARETO  = 0.05

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

def sample_data(n, pi, rng):
    n1 = int(round(pi * n)); n0 = n - n1
    n0p = int(round(0.5 * n0)); n0n = n0 - n0p
    X0  = np.vstack([rng.normal([3., 0.], 1., (n0p, 2)), rng.normal([0., 0.], 1., (n0n, 2))])
    n1p = int(round(0.5 * n1)); n1n = n1 - n1p
    X1  = np.vstack([rng.normal([0., 3.], 1., (n1p, 2)), rng.normal([0., 0.], 1., (n1n, 2))])
    X   = np.vstack([X0, X1])
    G   = np.concatenate([np.zeros(n0), np.ones(n1)])
    y   = np.concatenate([np.ones(n0p), np.zeros(n0n), np.ones(n1p), np.zeros(n1n)])
    perm = rng.permutation(len(y))
    return X[perm], G[perm].astype(int), y[perm].astype(int)

# ---------------------------------------------------------------------------
# Figure 1 — Phi curves (reference seed 0 + band over N_SEEDS)
# ---------------------------------------------------------------------------

print("=== Figure 1: Phi curves ===")

rng0 = np.random.default_rng(0)
X_tr, G_tr, y_tr = sample_data(N_TRAIN, PI_TRAIN, rng0)
X_te, G_te, y_te = sample_data(N_TEST,  0.5,      rng0)

erm = ERMBaseline(C=1.0); erm.fit(X_tr, G_tr, y_tr)
eg  = ExponentiatedGradient(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                             constraints=DemographicParity(), eps=0.01)
eg.fit(X_tr, y_tr, sensitive_features=G_tr)
socp = SOCPFairnessMixture(**SOCP_PARAMS); socp.fit(X_tr, G_tr, y_tr, seed=0)

p1_te = float(np.mean(G_te == 1))

def phi_curve(proba): return np.array([phi_dp(proba, G_te, pi, p1_te) for pi in PI_GRID])

phi_erm  = phi_curve(erm.predict_proba(X_te))
phi_eg   = phi_curve(eg_predict_proba(eg, X_te))
phi_socp = phi_curve(socp.predict_proba(X_te))

print(f"  Running {N_SEEDS} seeds for SOCP band...")
phi_socp_runs = []
for s in range(N_SEEDS):
    rng_s = np.random.default_rng(s)
    Xtr, Gtr, ytr = sample_data(N_TRAIN, PI_TRAIN, rng_s)
    Xte, Gte, yte = sample_data(N_TEST,  0.5,      rng_s)
    s_m = SOCPFairnessMixture(**SOCP_PARAMS); s_m.fit(Xtr, Gtr, ytr, seed=s)
    p1  = float(np.mean(Gte == 1))
    phi_socp_runs.append(np.array([phi_dp(s_m.predict_proba(Xte), Gte, pi, p1) for pi in PI_GRID]))
    print(f"    seed {s}: U(h)={s_m.unfairness_gap(Xte, Gte):.4f}", flush=True)

phi_socp_runs = np.array(phi_socp_runs)
phi_mean = phi_socp_runs.mean(0); phi_std = phi_socp_runs.std(0)

COLORS = {"erm": "#d62728", "eg": "#ff7f0e", "socp": "#1f77b4"}
u_erm  = erm.unfairness_gap(X_te, G_te)
u_eg   = abs(np.mean(eg_predict_proba(eg, X_te)[G_te==1]) - np.mean(eg_predict_proba(eg, X_te)[G_te==0]))
u_socp = socp.unfairness_gap(X_te, G_te)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(PI_GRID, phi_erm,  color=COLORS["erm"],  lw=2.0, label="ERM")
ax.plot(PI_GRID, phi_eg,   color=COLORS["eg"],   lw=2.0, label="ERM + DP (Agarwal et al.)")
ax.plot(PI_GRID, phi_mean, color=COLORS["socp"], lw=2.5, label="SOCP mixture (ours)")
ax.fill_between(PI_GRID, phi_mean-phi_std, phi_mean+phi_std,
                color=COLORS["socp"], alpha=0.15, label=r"SOCP $\pm$1 std")
ax.axhline(T_TARGET, color="black", lw=1.0, ls=":", alpha=0.7)
ax.text(1.015, T_TARGET, r"$t$", va="center", fontsize=12, transform=ax.get_yaxis_transform())
ax.axvline(PI_TRAIN, color="gray", lw=1.0, ls="--", alpha=0.4)
ax.text(PI_TRAIN+0.01, 0.27, r"$\pi_{\rm train}$", fontsize=9, color="gray")
for phi_c, color, label, yo in [
    (phi_erm,  COLORS["erm"],  fr"$U={u_erm:.2f}$",   -0.01),
    (phi_eg,   COLORS["eg"],   fr"$U={u_eg:.3f}$",    +0.01),
    (phi_mean, COLORS["socp"], fr"$U={u_socp:.3f}$",   0.00),
]:
    ax.annotate(label, xy=(1.0, float(phi_c[-1])), xytext=(1.03, float(phi_c[-1])+yo),
                fontsize=8.5, color=color, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
ax.set_xlabel(r"Deployment group proportion $\pi'$", fontsize=13)
ax.set_ylabel(r"$\Phi(h;\,\pi')$", fontsize=13)
ax.set_xlim(0.0, 1.0); ax.set_ylim(0.25, 0.75)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.legend(fontsize=10, loc="lower left"); ax.grid(True, alpha=0.2)
save_fig(fig, "synthetic_phi_curve", FIG_DIR)

# ---------------------------------------------------------------------------
# Figure 2 — Pareto frontiers (acc vs U, acc vs deploy, U vs deploy)
# ---------------------------------------------------------------------------

print("\n=== Figure 2: Pareto frontiers ===")

# ERM reference
erm_accs, erm_us, erm_gaps = [], [], []
for s in range(N_SEEDS):
    rng_s = np.random.default_rng(s)
    Xtr, Gtr, ytr = sample_data(N_TRAIN, PI_TRAIN, rng_s)
    Xte, Gte, yte = sample_data(N_TEST,  0.5,      rng_s)
    e = ERMBaseline(C=1.0); e.fit(Xtr, Gtr, ytr)
    p = e.predict_proba(Xte); p1 = float(np.mean(Gte==1))
    erm_accs.append(float(np.mean((p>=0.5)==yte)))
    erm_us.append(e.unfairness_gap(Xte, Gte))
    erm_gaps.append(max(abs(phi_dp(p,Gte,0.,p1)-T_TARGET), abs(phi_dp(p,Gte,1.,p1)-T_TARGET)))
erm_acc = np.mean(erm_accs); erm_u = np.mean(erm_us); erm_gap = np.mean(erm_gaps)

# SOCP sweep
results = []
for i, delta in enumerate(DELTA_GRID):
    accs, us, gaps = [], [], []
    for s in range(N_SEEDS):
        rng_s = np.random.default_rng(s)
        Xtr, Gtr, ytr = sample_data(N_TRAIN, PI_TRAIN, rng_s)
        Xte, Gte, yte = sample_data(N_TEST,  0.5,      rng_s)
        sm = SOCPFairnessMixture(K=30, pi_bar=PI_BAR_PARETO, m=60, delta=delta, t=T_TARGET, C=1.0)
        try:
            sm.fit(Xtr, Gtr, ytr, seed=i*N_SEEDS+s)
        except Exception: continue
        p = sm.predict_proba(Xte); p1 = float(np.mean(Gte==1))
        accs.append(float(np.mean((p>=0.5)==yte)))
        us.append(sm.unfairness_gap(Xte, Gte))
        gaps.append(max(abs(phi_dp(p,Gte,0.,p1)-T_TARGET), abs(phi_dp(p,Gte,1.,p1)-T_TARGET)))
        print(f"  delta={delta:.0e} seed={s}  U={us[-1]:.3f}", flush=True)
    if accs:
        results.append((delta, np.mean(accs), np.std(accs), np.mean(us), np.std(us),
                        np.mean(gaps), np.std(gaps)))

R = np.array(results)
LW, AB, FS = 3.5, 0.15, 18

for fname, x_mean, x_std, y_mean, y_std, xlabel, ylabel, fill_fn in [
    ("pareto_acc_vs_u",   R[:,3], R[:,4], R[:,1], R[:,2],
     r"$U(h)$ $\downarrow$", r"Accuracy $\uparrow$",
     lambda ax, xm, xs, ym, ys: ax.fill_betweenx(ym, xm-xs, xm+xs, color=COLORS["socp"], alpha=AB, linewidth=0)),
    ("pareto_acc_vs_gap", R[:,5], R[:,6], R[:,1], R[:,2],
     r"Deploy gap $\downarrow$", r"Accuracy $\uparrow$",
     lambda ax, xm, xs, ym, ys: ax.fill_betweenx(ym, xm-xs, xm+xs, color=COLORS["socp"], alpha=AB, linewidth=0)),
    ("pareto_u_vs_gap",   R[:,3], R[:,4], R[:,5], R[:,6],
     r"$U(h)$ $\downarrow$", r"Deploy gap $\downarrow$",
     lambda ax, xm, xs, ym, ys: ax.fill_between(xm, ym-ys, ym+ys, color=COLORS["socp"], alpha=AB, linewidth=0)),
]:
    erm_x = {"pareto_acc_vs_u": erm_u, "pareto_acc_vs_gap": erm_gap, "pareto_u_vs_gap": erm_u}[fname]
    erm_y = {"pareto_acc_vs_u": erm_acc,"pareto_acc_vs_gap": erm_acc,"pareto_u_vs_gap": erm_gap}[fname]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_mean, y_mean, color=COLORS["socp"], lw=LW, label=r"SOCP (sweep $\delta$)")
    fill_fn(ax, x_mean, x_std, y_mean, y_std)
    ax.scatter([erm_x], [erm_y], color=COLORS["erm"], s=120, zorder=5, label="ERM")
    ax.set_xlabel(xlabel, fontsize=FS); ax.set_ylabel(ylabel, fontsize=FS)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.2); ax.tick_params(labelsize=15)
    save_fig(fig, fname, FIG_DIR)

print("Done.")

# ---------------------------------------------------------------------------
# Figure 3 — Sensitivity to target t
# ---------------------------------------------------------------------------

print("\n=== Figure 3: Sensitivity to t ===")

T_GRID      = np.linspace(0.1, 0.9, 20)
DELTA_FIXED = 1e-3

# t naturel : niveau ERM moyen sur le test (seed 0)
rng0        = np.random.default_rng(0)
X_tr0, G_tr0, y_tr0 = sample_data(N_TRAIN, PI_TRAIN, rng0)
X_te0, G_te0, y_te0 = sample_data(N_TEST,  0.5,      rng0)
erm0        = ERMBaseline(C=1.0); erm0.fit(X_tr0, G_tr0, y_tr0)
p1_te0      = float(np.mean(G_te0 == 1))
T_NATURAL   = phi_dp(erm0.predict_proba(X_te0), G_te0, 0.5, p1_te0)
print(f"  t natural (ERM mean prediction) = {T_NATURAL:.3f}")

t_accs, t_us, t_gaps = [], [], []

for t_val in T_GRID:
    accs, us, gaps = [], [], []
    for s in range(N_SEEDS):
        rng_s = np.random.default_rng(s)
        Xtr, Gtr, ytr = sample_data(N_TRAIN, PI_TRAIN, rng_s)
        Xte, Gte, yte = sample_data(N_TEST,  0.5,      rng_s)
        sm = SOCPFairnessMixture(
            K=30, pi_bar=0.05, m=60,
            delta=DELTA_FIXED, t=t_val, C=1.0
        )
        try:
            sm.fit(Xtr, Gtr, ytr, seed=s)
        except Exception as e:
            print(f"    t={t_val:.2f} seed={s} FAILED: {e}")
            continue
        
        p = sm.predict_proba(Xte)
        p1 = float(np.mean(Gte == 1))
        phi_vals = sm.phi_curve(Xte, Gte, np.array([0.0, 1.0]))  # aux extremes

        accs.append(float(np.mean((p >= 0.5) == yte)))
        us.append(sm.unfairness_gap(Xte, Gte))
        gaps.append(max(
            abs(phi_vals[0] - t_val),
            abs(phi_vals[1] - t_val)
        ))
        print(f"    t={t_val:.2f} seed={s}  U={us[-1]:.3f}  gap={gaps[-1]:.3f}", flush=True)

    if accs:
        t_accs.append((np.mean(accs),  np.std(accs)))
        t_us.append(  (np.mean(us),    np.std(us)))
        t_gaps.append((np.mean(gaps),  np.std(gaps)))
    else:
        # SOCP infeasible pour ce t — ne devrait pas arriver grâce aux ConstantPredictors
        t_accs.append((np.nan, np.nan))
        t_us.append(  (np.nan, np.nan))
        t_gaps.append((np.nan, np.nan))

t_accs  = np.array(t_accs)
t_us    = np.array(t_us)
t_gaps  = np.array(t_gaps)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
labels = ["Accuracy $\\uparrow$", "$U(h)$ $\\downarrow$", "Deploy gap $\\downarrow$"]

for ax, vals, ylabel in zip(axes, [t_accs, t_us, t_gaps], labels):
    ax.plot(T_GRID, vals[:, 0], color=COLORS["socp"], lw=2.5)
    ax.fill_between(
        T_GRID,
        vals[:, 0] - vals[:, 1],
        vals[:, 0] + vals[:, 1],
        alpha=0.15, color=COLORS["socp"]
    )
    # t naturel ERM
    ax.axvline(T_NATURAL, color="black", ls=":", lw=1.2,
               label=f"$t_{{\\rm ERM}}={T_NATURAL:.2f}$")
    ax.set_xlabel("Target $t$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
save_fig(fig, "sensitivity_t", FIG_DIR)
print("Figure 3 saved.")