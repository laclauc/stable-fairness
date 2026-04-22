"""
Adult Income experiment — multi-seed evaluation.
Runs DP and EO constraints, saves results, generates figures, runs stat tests.

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

from utils import SOCPFairnessMixture, ERMBaseline
from postproc  import PostProcessor
from datasets.loaders import load_adult
from shared.evaluation import eg_predict_proba, compute_targets, eval_stats, make_phi_curve, test_stats
from shared.plotting import plot_phi_panel, save_fig, COLORS as PLOT_COLORS
from shared.stats_tests import run_wilcoxon_suite

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH  = ROOT / "datasets" / "adult" / "adult_norm.csv"
N_SEEDS    = 20
PI_GRID    = np.linspace(0.0, 1.0, 200)
RUN_SHIFTY = False

SOCP_DP_PARAMS = dict(K=30, pi_bar=0.02, m=60, delta=1e-5,  C=1.0, constraint="DP")
SOCP_EO_PARAMS = dict(K=30, pi_bar=0.15, m=60, delta=1e-5,  C=1.0, constraint="EO")

# PostProc (Chzhen, Hebiri & Taturyan, 2025) — post-processing baseline
PP_DP_PARAMS = dict(constraint="DP", eps=2**(-128), T=20_000, compute_sigma=False)
PP_EO_PARAMS = dict(constraint="EO", eps=2**(-128), T=20_000, compute_sigma=False)

# Shifty: import after utils to preserve sys.modules['utils']
if RUN_SHIFTY:
    from shifty import run_shifty_dp, run_shifty_eo

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

X_all, G_all, Y_all = load_adult(DATA_PATH)
print(f"Adult: n={len(Y_all)}, P(G=1)={G_all.mean():.3f}, P(Y=1)={Y_all.mean():.3f}")

# ---------------------------------------------------------------------------
# Multi-seed loop
# ---------------------------------------------------------------------------

rows    = []
curves  = {k: {"DP": [], "EO": []} for k in
           ["ERM", "EG_DP", "SOCP_DP", "PP_DP", "EG_EO", "SOCP_EO", "PP_EO"]}
if RUN_SHIFTY:
    curves["Shifty_DP"] = {"DP": []}
    curves["Shifty_EO"] = {"EO": []}
t_dp_list, t_eo_list = [], []

t_total = time.time()

for seed in range(N_SEEDS):
    print(f"\n=== Seed {seed+1}/{N_SEEDS} ===", flush=True)

    X_tr, X_te, G_tr, G_te, y_tr, y_te = train_test_split(
        X_all, G_all, Y_all, test_size=0.3, random_state=seed
    )
    p1, p11, p01 = test_stats(G_te, y_te)

    # ERM
    erm = ERMBaseline(C=1.0)
    erm.fit(X_tr, G_tr, y_tr)
    erm_p       = erm.predict_proba(X_te)
    t_dp, t_eo  = compute_targets(erm, X_tr, G_tr, y_tr)
    t_dp_list.append(t_dp); t_eo_list.append(t_eo)

    for c, t in [("DP", t_dp), ("EO", t_eo)]:
        acc, u, dep = eval_stats(erm_p, G_te, y_te, t, p1, p11, p01, c)
        rows.append(dict(model="ERM", constraint=c, seed=seed, acc=acc, u=u, deploy=dep, time=0))
        curves["ERM"][c].append(make_phi_curve(erm_p, G_te, y_te, PI_GRID, p1, p11, p01, c))
    print(f"  ERM   DP U={eval_stats(erm_p,G_te,y_te,t_dp,p1,p11,p01,'DP')[1]:.3f}  "
          f"EO U={eval_stats(erm_p,G_te,y_te,t_eo,p1,p11,p01,'EO')[1]:.3f}", flush=True)

    # EG DP
    t0 = time.time()
    eg_dp = ExponentiatedGradient(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                                   constraints=DemographicParity(), eps=0.01)
    eg_dp.fit(X_tr, y_tr, sensitive_features=G_tr)
    eg_dp_p = eg_predict_proba(eg_dp, X_te)
    acc, u, dep = eval_stats(eg_dp_p, G_te, y_te, t_dp, p1, p11, p01, "DP")
    rows.append(dict(model="EG_DP", constraint="DP", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["EG_DP"]["DP"].append(make_phi_curve(eg_dp_p, G_te, y_te, PI_GRID, p1, p11, p01, "DP"))
    print(f"  EG_DP   U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # SOCP DP
    t0 = time.time()
    socp_dp = SOCPFairnessMixture(**SOCP_DP_PARAMS, t=t_dp)
    try:
        socp_dp.fit(X_tr, G_tr, y_tr, seed=seed)
        socp_dp_p = socp_dp.predict_proba(X_te)
        acc, u, dep = eval_stats(socp_dp_p, G_te, y_te, t_dp, p1, p11, p01, "DP")
    except RuntimeError:
        socp_dp_p = erm_p
        acc, u, dep = np.nan, np.nan, np.nan
        print(f"  SOCP_DP INFEASIBLE", flush=True)
    rows.append(dict(model="SOCP_DP", constraint="DP", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["SOCP_DP"]["DP"].append(make_phi_curve(socp_dp_p, G_te, y_te, PI_GRID, p1, p11, p01, "DP"))
    print(f"  SOCP_DP U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # PostProc DP — split TRAINING data, fit estimators on train_, post-process on unlab
    t0 = time.time()
    try:
        X_train_, X_unlab, G_train_, G_unlab, y_train_, y_unlab = train_test_split(
            X_tr, G_tr, y_tr, test_size=0.5, random_state=seed,
        )
        base_pp = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_train_, y_train_)
        # tau and marginals: use ALL training data (G is available everywhere at fit-time)
        tau_pp  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_tr, G_tr)
        p_S_pp  = np.array([float(np.mean(G_tr == 0)), float(np.mean(G_tr == 1))])
        post = PostProcessor(
            base_estimator=base_pp, tau_estimator=tau_pp, p_S=p_S_pp,
            **PP_DP_PARAMS,
        )
        post.fit(X_unlab, seed=seed)
        pp_dp_p = post.predict_proba(X_te)
        acc, u, dep = eval_stats(pp_dp_p, G_te, y_te, t_dp, p1, p11, p01, "DP")
    except Exception as e:
        pp_dp_p = erm_p; acc, u, dep = np.nan, np.nan, np.nan
        print(f"  PP_DP ERROR: {e}", flush=True)
    rows.append(dict(model="PP_DP", constraint="DP", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["PP_DP"]["DP"].append(make_phi_curve(pp_dp_p, G_te, y_te, PI_GRID, p1, p11, p01, "DP"))
    print(f"  PP_DP U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # EG EO
    t0 = time.time()
    eg_eo = ExponentiatedGradient(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
                                   constraints=EqualizedOdds(), eps=0.01)
    eg_eo.fit(X_tr, y_tr, sensitive_features=G_tr)
    eg_eo_p = eg_predict_proba(eg_eo, X_te)
    acc, u, dep = eval_stats(eg_eo_p, G_te, y_te, t_eo, p1, p11, p01, "EO")
    rows.append(dict(model="EG_EO", constraint="EO", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["EG_EO"]["EO"].append(make_phi_curve(eg_eo_p, G_te, y_te, PI_GRID, p1, p11, p01, "EO"))
    print(f"  EG_EO   U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # SOCP EO
    t0 = time.time()
    socp_eo = SOCPFairnessMixture(**SOCP_EO_PARAMS, t=t_eo)
    try:
        socp_eo.fit(X_tr, G_tr, y_tr, seed=seed)
        socp_eo_p = socp_eo.predict_proba(X_te)
        acc, u, dep = eval_stats(socp_eo_p, G_te, y_te, t_eo, p1, p11, p01, "EO")
    except RuntimeError:
        socp_eo_p = erm_p
        acc, u, dep = np.nan, np.nan, np.nan
        print(f"  SOCP_EO INFEASIBLE", flush=True)
    rows.append(dict(model="SOCP_EO", constraint="EO", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["SOCP_EO"]["EO"].append(make_phi_curve(socp_eo_p, G_te, y_te, PI_GRID, p1, p11, p01, "EO"))
    print(f"  SOCP_EO U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # PostProc EO — split TRAINING data, fit base/tau/q on train_, post-process on unlab
    t0 = time.time()
    try:
        X_train_, X_unlab, G_train_, G_unlab, y_train_, y_unlab = train_test_split(
            X_tr, G_tr, y_tr, test_size=0.5, random_state=seed,
        )
        base_pp = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_train_, y_train_)
        # tau, joint, and marginals: use ALL training data (G and y are available at fit-time)
        tau_pp  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_tr, G_tr)
        # joint_estimator = list of binary classifiers; column s estimates P((S,Y)=(s,1) | X).
        joint_pp = [
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(
                X_tr, ((G_tr == s) & (y_tr == 1)).astype(int)
            )
            for s in (0, 1)
        ]
        p_S_pp    = np.array([float(np.mean(G_tr == 0)), float(np.mean(G_tr == 1))])
        p_Y1_pp   = float(np.mean(y_tr == 1))
        p_S_Y1_pp = np.array([float(np.mean((G_tr == 0) & (y_tr == 1))),
                              float(np.mean((G_tr == 1) & (y_tr == 1)))])
        post = PostProcessor(
            base_estimator=base_pp, tau_estimator=tau_pp, joint_estimator=joint_pp,
            p_S=p_S_pp, p_Y1=p_Y1_pp, p_S_Y1=p_S_Y1_pp,
            **PP_EO_PARAMS,
        )
        post.fit(X_unlab, seed=seed)
        pp_eo_p = post.predict_proba(X_te)
        acc, u, dep = eval_stats(pp_eo_p, G_te, y_te, t_eo, p1, p11, p01, "EO")
    except Exception as e:
        pp_eo_p = erm_p; acc, u, dep = np.nan, np.nan, np.nan
        print(f"  PP_EO ERROR: {e}", flush=True)
    rows.append(dict(model="PP_EO", constraint="EO", seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
    curves["PP_EO"]["EO"].append(make_phi_curve(pp_eo_p, G_te, y_te, PI_GRID, p1, p11, p01, "EO"))
    print(f"  PP_EO U={u:.3f} dep={dep:.3f} acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # Shifty DP + EO
    if RUN_SHIFTY:
        for tag, run_fn, c, t, params in [
            ("Shifty_DP", run_shifty_dp, "DP", t_dp, dict(epsilon_fairness=0.10, setting=2, alpha=0.25, seed=seed)),
            ("Shifty_EO", run_shifty_eo, "EO", t_eo, dict(epsilon_fairness=0.15, setting=2, alpha=0.25, seed=seed)),
        ]:
            t0 = time.time()
            try:
                sh   = run_fn(X_tr, G_tr, y_tr, **params)
                sh_p = sh.predict_proba(X_te)
                acc, u, dep = eval_stats(sh_p, G_te, y_te, t, p1, p11, p01, c)
                nsf = not sh.accepted
            except Exception as e:
                sh_p = erm_p; acc, u, dep, nsf = np.nan, np.nan, np.nan, True
                print(f"  {tag} ERROR: {e}", flush=True)
            rows.append(dict(model=tag, constraint=c, seed=seed, acc=acc, u=u, deploy=dep, time=time.time()-t0))
            curves[tag][c].append(make_phi_curve(sh_p, G_te, y_te, PI_GRID, p1, p11, p01, c))
            print(f"  {tag:<11} U={u:.3f} dep={dep:.3f} acc={acc:.3f} NSF={nsf}  ({time.time()-t0:.1f}s)", flush=True)

print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df = pd.DataFrame(rows)
df.to_csv(RES_DIR / "adult_multiseed_results.csv", index=False)
np.save(RES_DIR / "adult_phi_curves_DP.npy",
        {"PI_GRID": PI_GRID, "ERM": np.array(curves["ERM"]["DP"]),
         "EG_DP": np.array(curves["EG_DP"]["DP"]), "SOCP_DP": np.array(curves["SOCP_DP"]["DP"]),
         "PP_DP": np.array(curves["PP_DP"]["DP"]),
         "t_dp": np.array(t_dp_list),
         **({} if not RUN_SHIFTY else {"Shifty_DP": np.array(curves["Shifty_DP"]["DP"])})})
np.save(RES_DIR / "adult_phi_curves_EO.npy",
        {"PI_GRID": PI_GRID, "ERM": np.array(curves["ERM"]["EO"]),
         "EG_EO": np.array(curves["EG_EO"]["EO"]), "SOCP_EO": np.array(curves["SOCP_EO"]["EO"]),
         "PP_EO": np.array(curves["PP_EO"]["EO"]),
         "t_eo": np.array(t_eo_list),
         **({} if not RUN_SHIFTY else {"Shifty_EO": np.array(curves["Shifty_EO"]["EO"])})})
print("Results saved.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("SUMMARY — Adult Income (mean ± std)")
print("="*80)
for c, models in [("DP", ["ERM","EG_DP","SOCP_DP","PP_DP"]),
                  ("EO", ["ERM","EG_EO","SOCP_EO","PP_EO"])]:
    if RUN_SHIFTY:
        models += [f"Shifty_{'DP' if c=='DP' else 'EO'}"]
    print(f"\n--- {c} ---")
    for m in models:
        sub = df[(df.model == m) & (df.constraint == c)]
        print(f"  {m:<12}  U={sub.u.mean():.3f}±{sub.u.std():.3f}  "
              f"deploy={sub.deploy.mean():.3f}±{sub.deploy.std():.3f}  "
              f"acc={sub.acc.mean():.3f}±{sub.acc.std():.3f}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

dp = np.load(RES_DIR / "adult_phi_curves_DP.npy", allow_pickle=True).item()
eo = np.load(RES_DIR / "adult_phi_curves_EO.npy", allow_pickle=True).item()

for tag, data, c_key, t_key, eg_key, socp_key, pp_key in [
    ("adult_DP", dp, "DP", "t_dp", "EG_DP",  "SOCP_DP", "PP_DP"),
    ("adult_EO", eo, "EO", "t_eo", "EG_EO",  "SOCP_EO", "PP_EO"),
]:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    extra = [(data[pp_key], PLOT_COLORS["PostProc"], "PostProc")]
    plot_phi_panel(ax, data["ERM"], data[eg_key], data[socp_key],
                   data["PI_GRID"], float(np.mean(data[t_key])),
                   extra_curves=extra)
    save_fig(fig, tag, FIG_DIR)

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

#run_wilcoxon_suite(df, [
#    ("SOCP_DP", "EG_DP", "DP"),
#    ("SOCP_EO", "EG_EO", "EO"),
#])
