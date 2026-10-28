"""ACS Income: train on California, evaluate on 11 held-out states.

The held-out states differ in their racial composition, i.e. in a real (not
simulated) demographic shift. Each (method, budget, seed) is trained once on
CA and evaluated on every state; one row per state is written to

    results/acs_pareto_<constraint><tag>.csv

Per state: pi = P(G=1) for DP and P(G=1 | Y=1) for EO; Phi is the mean
prediction (on Y=1 for EO); deploy = |Phi - t|. The same file feeds both the
geographic figures and the ACS table (plots/tables.py).

Usage
-----
    python experiments/acs_pareto.py --constraint DP --seeds 5
    python experiments/acs_pareto.py --constraint EO --seeds 5
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from methods import DELTA0_GRID, FACTORIES, LAM_GRID, RESULTS, SELECTED, DROSweep, parse_methods
from stable import ERMBaseline
from stable.data import ACS_TEST_STATES, ACS_TRAIN_STATE, load_acs, stratified_subsample
from stable.metrics import accuracy, targets, unfairness

warnings.filterwarnings("ignore")

TRAIN_N, TEST_N = 80_000, 30_000

p = argparse.ArgumentParser()
p.add_argument("--constraint", required=True, choices=["DP", "EO"])
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--seed-start", type=int, default=0,
               help="first seed (to complete an earlier run)")
p.add_argument("--methods", default=None, help="default: STABLE,EG,DRO")
p.add_argument("--grid", default=None, help="comma-separated delta0 values")
p.add_argument("--lam-grid", default=None, help="comma-separated lambda values (STABLE_DRO)")
p.add_argument("--rho", type=float, default=0.9)
p.add_argument("--tag", default="")
args = p.parse_args()

C = args.constraint
methods = parse_methods(args.methods, "STABLE,EG,DRO")
grid = [float(v) for v in args.grid.split(",")] if args.grid else DELTA0_GRID["acs"]
lam_grid = [float(v) for v in args.lam_grid.split(",")] if args.lam_grid else LAM_GRID[("acs", C)]
out = RESULTS / f"acs_pareto_{C}{args.tag}.csv"
out.parent.mkdir(exist_ok=True)

X_ca, G_ca, y_ca = load_acs(ACS_TRAIN_STATE)
states = {s: load_acs(s, max_n=TEST_N, seed=0) for s in ACS_TEST_STATES}
print(f"{ACS_TRAIN_STATE}: n={len(y_ca)} | {len(states)} test states | methods={methods}")


def state_metrics(h, G, y, t):
    if C == "DP":
        pi, phi = float(np.mean(G == 1)), float(np.mean(h))
    else:
        pos = y == 1
        pi, phi = float(np.mean(G[pos] == 1)), float(np.mean(h[pos]))
    return dict(pi=pi, phi=phi, t=t, deploy=abs(phi - t),
                acc=accuracy(h, y), u=unfairness(h, G, y, C))


sel = SELECTED[("acs", C)]          # pi_bar and dictionary, selected by tuning.py
print(f"pi_bar={sel['pi_bar']}  label-flipped components={sel['add_counterfactual']}")
rows, t_start = [], time.time()
for seed in range(args.seed_start, args.seeds):
    Xtr, Gtr, ytr = stratified_subsample(X_ca, G_ca, y_ca, TRAIN_N, seed)
    erm = ERMBaseline().fit(Xtr, Gtr, ytr)
    t_dp, t_eo = targets(erm.predict_proba(Xtr), Gtr, ytr)
    t = t_dp if C == "DP" else t_eo

    def record(method, knob, value, model, elapsed):
        base = dict(method=method, knob=knob, delta0=value, seed=seed, time=elapsed)
        if model is None:
            rows.append(base | dict(state=None, ok=False))
            return
        for s, (Xs, Gs, ys) in states.items():
            rows.append(base | dict(state=s, ok=True,
                                    **state_metrics(model.predict_proba(Xs), Gs, ys, t)))

    record("ERM", None, np.nan, erm, 0.0)
    for name in methods:
        if name == "DRO":
            t0 = time.time()
            sweep = DROSweep(Xtr, Gtr, ytr, t, C, seed, rho=args.rho, **sel)
            setup = time.time() - t0
            for lam in lam_grid:
                t0 = time.time()
                record("DRO", "lambda", lam, sweep(lam), time.time() - t0 + setup)
                setup = 0.0
            continue
        for delta0 in grid:
            t0 = time.time()
            try:
                kw = ({"T": 25_000} if name == "PostProc"
                      else sel if name.startswith("STABLE") else {})
                model = FACTORIES[name](Xtr, Gtr, ytr, t, delta0, C, seed, **kw)
            except Exception as e:
                print(f"  {name} delta0={delta0} seed={seed}: {type(e).__name__}: {e}")
                model = None
            record(name, "delta0", delta0, model, time.time() - t0)
            print(f"  seed {seed} {name} delta0={delta0} ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"seed {seed} done ({(time.time() - t_start) / 60:.0f} min)", flush=True)

print(f"-> {out}")
