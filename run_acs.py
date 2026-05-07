"""
ACS geographic robustness experiment — multi-seed evaluation.
Trains on CA, evaluates on multiple states. DP constraint only.

Config: edit the constants at the top.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from folktables import ACSDataSource, ACSIncome

from utils import SOCPFairnessMixture, ERMBaseline, phi_dp, unfairness_gap_dp
from postproc  import PostProcessor
from shared.evaluation import eg_predict_proba
from shared.plotting import save_fig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SURVEY_YEAR = "2018"
TRAIN_STATE = "CA"
N_SEEDS     = 5
TRAIN_N     = 80_000
EVAL_N      = 30_000
RUN_SHIFTY  = True

SOCP_PARAMS = dict(K=30, pi_bar=0.02, m=60, delta=1e-5, C=1.0,
                   task="classification", constraint="DP")
PP_PARAMS = dict(constraint="DP", eps=2**(-128), T=25_000, compute_sigma=False)

if RUN_SHIFTY:
    from shifty import run_shifty_dp

EVAL_STATES = {
    "ME": "Maine",      "WA": "Washington",  "NY": "New York",
    "FL": "Florida",    "WI": "Wisconsin",   "IL": "Illinois",
    "PA": "Pennsylvania","OH": "Ohio",       "MI": "Michigan",
    "GA": "Georgia",    "OR": "Oregon",
}

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

print("Loading ACS data...")
src = ACSDataSource(survey_year=SURVEY_YEAR, horizon="1-Year", survey="person")

X_tr_full, y_tr_full, G_tr_raw = ACSIncome.df_to_numpy(
    src.get_data(states=[TRAIN_STATE], download=True)
)
y_tr_full = y_tr_full.astype(int)
G_tr_full = (G_tr_raw == 1).astype(int)
print(f"{TRAIN_STATE}: n={len(y_tr_full)}, P(G=1)={G_tr_full.mean():.3f}")

eval_data = {}
for code in EVAL_STATES:
    X_s, y_s, G_s_raw = ACSIncome.df_to_numpy(src.get_data(states=[code], download=True))
    y_s = y_s.astype(int); G_s = (G_s_raw == 1).astype(int)
    if len(y_s) > EVAL_N:
        _, idx = train_test_split(np.arange(len(y_s)), test_size=EVAL_N/len(y_s),
                                   random_state=0, stratify=np.stack([y_s, G_s], axis=1))
        X_s, y_s, G_s = X_s[idx], y_s[idx], G_s[idx]
    eval_data[code] = (X_s, G_s, y_s)
    print(f"  {code}: n={len(y_s)}, pi={G_s.mean():.3f}")

# ---------------------------------------------------------------------------
# Multi-seed loop
# ---------------------------------------------------------------------------

def metrics(proba, G_s, y_s, t):
    acc    = float(np.mean((proba >= 0.5).astype(int) == y_s))
    u      = unfairness_gap_dp(proba, G_s)
    pi     = float(np.mean(G_s == 1))
    phi_v  = phi_dp(proba, G_s, pi, pi)
    deploy = abs(phi_v - t)
    return acc, u, phi_v, deploy

all_rows = []
t_total  = time.time()

for seed in range(N_SEEDS):
    print(f"\n=== Seed {seed} ===", flush=True)

    n = min(TRAIN_N, len(y_tr_full))
    _, idx = train_test_split(np.arange(len(y_tr_full)), test_size=n/len(y_tr_full),
                               random_state=seed,
                               stratify=np.stack([y_tr_full, G_tr_full], axis=1))
    X_tr = X_tr_full[idx]; G_tr = G_tr_full[idx]; y_tr = y_tr_full[idx]

    t0 = time.time()
    erm = ERMBaseline(C=1.0, task="classification")
    erm.fit(X_tr, G_tr, y_tr)
    t_target = float(np.mean(erm.predict_proba(X_tr)))
    print(f"  ERM     {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    eg = ExponentiatedGradient(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                                constraints=DemographicParity(), eps=0.01)
    eg.fit(X_tr, y_tr, sensitive_features=G_tr)
    print(f"  EG      {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    socp = SOCPFairnessMixture(**SOCP_PARAMS, t=t_target)
    try:
        socp.fit(X_tr, G_tr, y_tr, seed=seed); socp_ok = True
        print(f"  SOCP    {time.time()-t0:.1f}s", flush=True)
    except RuntimeError:
        socp_ok = False; print("  SOCP INFEASIBLE")

    t0 = time.time()
    try:
        # Split TRAINING data into (train_, unlab) for PostProc
        X_train_, X_unlab, G_train_, G_unlab, y_train_, y_unlab = train_test_split(
            X_tr, G_tr, y_tr, test_size=0.5, random_state=seed,
        )
        base_pp = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_train_, y_train_)
        # tau and marginals: use ALL training data (G is available everywhere at fit-time)
        tau_pp  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_tr, G_tr)
        p_S_pp  = np.array([float(np.mean(G_tr == 0)), float(np.mean(G_tr == 1))])
        post = PostProcessor(
            base_estimator=base_pp, tau_estimator=tau_pp, p_S=p_S_pp,
            **PP_PARAMS,
        )
        post.fit(X_unlab, seed=seed); pp_ok = True
        print(f"  PostProc    {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        pp_ok = False; print(f"  PostProc ERROR: {e}")

    shifty_ok = False
    if RUN_SHIFTY:
        t0 = time.time()
        try:
            sh = run_shifty_dp(X_tr, G_tr, y_tr, epsilon_fairness=0.05,
                               setting=2, alpha=0.25, seed=seed)
            shifty_ok = sh.accepted
            print(f"  Shifty  {time.time()-t0:.1f}s  NSF={not sh.accepted}", flush=True)
        except Exception as e:
            print(f"  Shifty ERROR: {e}", flush=True)

    for code, (X_s, G_s, y_s) in eval_data.items():
        erm_p    = erm.predict_proba(X_s)
        eg_p     = eg_predict_proba(eg, X_s)
        socp_p   = socp.predict_proba(X_s)             if socp_ok   else erm_p
        pp_p   = post.predict_proba(X_s)      if pp_ok   else erm_p
        shifty_p = sh.predict_proba(X_s)               if shifty_ok else erm_p
        pi       = float(np.mean(G_s == 1))

        acc_erm,    u_erm,    phi_erm,    dep_erm    = metrics(erm_p,    G_s, y_s, t_target)
        acc_eg,     u_eg,     phi_eg,     dep_eg     = metrics(eg_p,     G_s, y_s, t_target)
        acc_socp,   u_socp,   phi_socp,   dep_socp   = metrics(socp_p,   G_s, y_s, t_target)
        acc_pp,   u_pp,   phi_pp,   dep_pp   = metrics(pp_p,   G_s, y_s, t_target)
        acc_shifty, u_shifty, phi_shifty, dep_shifty = metrics(shifty_p, G_s, y_s, t_target)

        all_rows.append(dict(
            seed=seed, state=code, pi=pi, t=t_target,
            phi_erm=phi_erm,       u_erm=u_erm,       acc_erm=acc_erm,       dep_erm=dep_erm,
            phi_eg=phi_eg,         u_eg=u_eg,         acc_eg=acc_eg,         dep_eg=dep_eg,
            phi_socp=phi_socp,     u_socp=u_socp,     acc_socp=acc_socp,     dep_socp=dep_socp,
            phi_pp=phi_pp,     u_pp=u_pp,     acc_pp=acc_pp,     dep_pp=dep_pp,
            phi_shifty=phi_shifty, u_shifty=u_shifty, acc_shifty=acc_shifty, dep_shifty=dep_shifty,
        ))
        print(f"  {code} pi={pi:.3f} | ERM dep={dep_erm:.3f} | EG dep={dep_eg:.3f} | "
              f"SOCP dep={dep_socp:.3f} | PostProc dep={dep_pp:.3f} | Shifty dep={dep_shifty:.3f}",
              flush=True)

print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

# ---------------------------------------------------------------------------
# Save (append if file exists for easy multi-run accumulation)
# ---------------------------------------------------------------------------

csv_path = RES_DIR / "acs_geographic_robustness.csv"
df_new   = pd.DataFrame(all_rows)
if csv_path.exists():
    df_old  = pd.read_csv(csv_path)
    if "seed" not in df_old.columns: df_old["seed"] = -1
    df_new["seed"] += int(df_old["seed"].max()) + 1
    df_all  = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_all = df_new
df_all.to_csv(csv_path, index=False)
print(f"Saved {len(df_all)} rows to {csv_path}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

df = df_all.groupby("state").agg(
    pi=("pi","mean"), t=("t","mean"),
    phi_erm=("phi_erm","mean"),       phi_erm_std=("phi_erm","std"),
    phi_eg=("phi_eg","mean"),         phi_eg_std=("phi_eg","std"),
    phi_socp=("phi_socp","mean"),     phi_socp_std=("phi_socp","std"),
    phi_pp=("phi_pp","mean"),     phi_pp_std=("phi_pp","std"),
    phi_shifty=("phi_shifty","mean"), phi_shifty_std=("phi_shifty","std"),
    dep_erm=("dep_erm","mean"),       dep_erm_std=("dep_erm","std"),
    dep_eg=("dep_eg","mean"),         dep_eg_std=("dep_eg","std"),
    dep_socp=("dep_socp","mean"),     dep_socp_std=("dep_socp","std"),
    dep_pp=("dep_pp","mean"),     dep_pp_std=("dep_pp","std"),
    dep_shifty=("dep_shifty","mean"), dep_shifty_std=("dep_shifty","std"),
    u_erm=("u_erm","mean"),   u_eg=("u_eg","mean"),
    u_socp=("u_socp","mean"), u_pp=("u_pp","mean"), u_shifty=("u_shifty","mean"),
    acc_erm=("acc_erm","mean"),   acc_eg=("acc_eg","mean"),
    acc_socp=("acc_socp","mean"), acc_pp=("acc_pp","mean"), acc_shifty=("acc_shifty","mean"),
).reset_index().sort_values("pi")

t_mean  = float(df["t"].mean())
COLORS  = {"erm": "#d62728", "eg": "#ff7f0e", "socp": "#1f77b4",
           "pp": "#2ca02c", "shifty": "#9467bd"}
FS_L, FS_T, FS_K = 18, 15, 16

# Scatter: Phi vs pi_state
fig, ax = plt.subplots(figsize=(7, 5))
for col, col_std, color, marker, label in [
    ("phi_erm",    "phi_erm_std",    COLORS["erm"],    "o", "ERM"),
    ("phi_eg",     "phi_eg_std",     COLORS["eg"],     "s", "ERM + DP"),
    ("phi_socp",   "phi_socp_std",   COLORS["socp"],   "^", "Ours"),
    ("phi_pp",   "phi_pp_std",   COLORS["pp"],   "v", "PostProc"),
    ("phi_shifty", "phi_shifty_std", COLORS["shifty"], "D", "Shifty"),
]:
    ax.scatter(df["pi"], df[col], color=color, marker=marker, s=120, zorder=4, label=label)
    ax.errorbar(df["pi"], df[col], yerr=df[col_std], fmt="none", color=color, capsize=4, lw=1.5)
    z = np.polyfit(df["pi"], df[col], 1)
    pi_l = np.linspace(df["pi"].min()-0.02, df["pi"].max()+0.02, 100)
    ax.plot(pi_l, np.poly1d(z)(pi_l), color=color, lw=2, ls="--", alpha=0.5)
for _, row in df.iterrows():
    ax.annotate(row["state"], xy=(row["pi"], row["phi_socp"]),
                xytext=(5,5), textcoords="offset points", fontsize=11, color=COLORS["socp"])
ax.axhline(t_mean, color="black", lw=1.2, ls=":", alpha=0.7)
ax.text(1.015, t_mean, r"$t$", va="center", fontsize=FS_L, transform=ax.get_yaxis_transform())
ax.set_xlabel(r"State proportion of White residents $\pi_{\rm state}$", fontsize=FS_L)
ax.set_ylabel(r"$\Phi(h;\,\pi_{\rm state})$", fontsize=FS_L)
ax.tick_params(labelsize=FS_T); ax.legend(fontsize=FS_K, loc="upper left", ncol=2); ax.grid(True, alpha=0.2)
save_fig(fig, "acs_geo_scatter", FIG_DIR)

# Barplot: deploy gap per state
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df)); w = 0.16
for offset, col, col_std, color, label in [
    (-2*w, "dep_erm",    "dep_erm_std",    COLORS["erm"],    "ERM"),
    (-1*w, "dep_eg",     "dep_eg_std",     COLORS["eg"],     "ERM + DP"),
    ( 0*w, "dep_socp",   "dep_socp_std",   COLORS["socp"],   "Ours"),
    ( 1*w, "dep_pp",   "dep_pp_std",   COLORS["pp"],   "PostProc"),
    ( 2*w, "dep_shifty", "dep_shifty_std", COLORS["shifty"], "Shifty"),
]:
    ax.bar(x+offset, df[col], width=w, color=color, label=label, alpha=0.88)
    ax.errorbar(x+offset, df[col], yerr=df[col_std], fmt="none", color="black", capsize=3, lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(df["state"].values, fontsize=FS_T)
ax.set_ylabel(r"$|\Phi(h;\,\pi_{\rm state}) - t|$", fontsize=FS_L)
ax.tick_params(labelsize=FS_T); ax.legend(fontsize=FS_K, loc="upper left", ncol=3)
ax.grid(True, alpha=0.2, axis="y"); ax.set_ylim(bottom=0)
save_fig(fig, "acs_geo_barplot", FIG_DIR)
