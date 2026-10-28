"""Pareto frontiers on Adult / COMPAS (main-text figure and appendix).

Each method is swept over the common fairness-budget grid delta0 (STABLE_DRO
over its own lambda grid), for several random 70/30 train/test splits. One row
per (method, budget, seed) is written to

    results/pareto_<dataset>_<constraint><tag>.csv

Usage
-----
    python experiments/pareto.py --dataset adult  --constraint DP --seeds 10
    python experiments/pareto.py --dataset compas --constraint EO --seeds 10
    # Shifty and PostProc, if their code is on the Python path (see README)
    python experiments/pareto.py --dataset adult --constraint DP --methods PostProc,Shifty --tag _extra
    # dictionary ablation (appendix)
    python experiments/pareto.py --dataset adult --constraint DP \
        --methods STABLE_CF,STABLE_noCF --tag _cf_ablation
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from methods import DELTA0_GRID, FACTORIES, LAM_GRID, RESULTS, SELECTED, DROSweep, parse_methods
from stable import ERMBaseline
from stable.data import load
from stable.metrics import evaluate, targets, unfairness, worst_case_accuracy

warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["adult", "compas"])
p.add_argument("--constraint", required=True, choices=["DP", "EO"])
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--methods", default=None, help="default: STABLE,EG,DRO")
p.add_argument("--grid", default=None, help="comma-separated delta0 values")
p.add_argument("--lam-grid", default=None, help="comma-separated lambda values (STABLE_DRO)")
p.add_argument("--rho", type=float, default=0.9, help="STABLE_DRO radius")
p.add_argument("--tag", default="", help="suffix of the output file")
args = p.parse_args()

C = args.constraint
methods = parse_methods(args.methods, "STABLE,EG,DRO")
grid = [float(v) for v in args.grid.split(",")] if args.grid else DELTA0_GRID[args.dataset]
lam_grid = [float(v) for v in args.lam_grid.split(",")] if args.lam_grid else LAM_GRID[(args.dataset, C)]
out = RESULTS / f"pareto_{args.dataset}_{C}{args.tag}.csv"
out.parent.mkdir(exist_ok=True)

X_all, G_all, y_all = load(args.dataset)
print(f"{args.dataset}/{C}: n={len(y_all)}  methods={methods}")

sel = SELECTED[(args.dataset, C)]          # pi_bar and dictionary, selected by tuning.py
print(f"pi_bar={sel['pi_bar']}  label-flipped components={sel['add_counterfactual']}")
rows, t_start = [], time.time()
for seed in range(args.seeds):
    Xtr, Xte, Gtr, Gte, ytr, yte = train_test_split(
        X_all, G_all, y_all, test_size=0.3, random_state=seed)
    erm = ERMBaseline().fit(Xtr, Gtr, ytr)
    t_dp, t_eo = targets(erm.predict_proba(Xtr), Gtr, ytr)
    t = t_dp if C == "DP" else t_eo

    def record(method, knob, value, model, elapsed):
        row = dict(method=method, knob=knob, delta0=value, seed=seed, time=elapsed)
        if model is None:
            rows.append(row | dict(ok=False))
            return
        h = model.predict_proba(Xte)
        rows.append(row | dict(ok=True, **evaluate(h, Gte, yte, t, C),
                               u_train=unfairness(model.predict_proba(Xtr), Gtr, ytr, C),
                               acc_worst=worst_case_accuracy(h, Gte, yte, C, sel["pi_bar"])))

    record("ERM", None, np.nan, erm, 0.0)

    for name in methods:
        if name == "DRO":
            t0 = time.time()
            sweep = DROSweep(Xtr, Gtr, ytr, t, C, seed, rho=args.rho, **sel)
            setup = time.time() - t0          # dictionary fit, shared by all lambdas
            for lam in lam_grid:
                t0 = time.time()
                record("DRO", "lambda", lam, sweep(lam), time.time() - t0 + setup)
                setup = 0.0
            continue
        for delta0 in grid:
            t0 = time.time()
            try:
                kw = sel if name.startswith("STABLE") else {}
                model = FACTORIES[name](Xtr, Gtr, ytr, t, delta0, C, seed, **kw)
            except Exception as e:  # infeasible SOCP, abstention, ...
                print(f"  {name} delta0={delta0} seed={seed}: {type(e).__name__}: {e}")
                model = None
            record(name, "delta0", delta0, model, time.time() - t0)

    pd.DataFrame(rows).to_csv(out, index=False)          # checkpoint every seed
    print(f"  seed {seed} done ({time.time() - t_start:.0f}s)", flush=True)

print(f"-> {out}")
