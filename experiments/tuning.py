"""Hyperparameter selection for STABLE on one validation split.

Adult / COMPAS: 70/30 split with seed 0, metrics on the 30% held-out part.
ACS: train on CA, metrics averaged over validation states disjoint from the
test states.

Two choices are made on the validation data, with rules fixed in advance:

1. pi_bar: grid over (pi_bar, delta), for each dictionary (with and without
   the label-flipped components); pi_bar is taken from the point with the
   smallest deployment gap. This grid also gives the sensitivity heatmaps.
2. dictionary: the deployment gap does not separate the two dictionaries, so
   they are compared on accuracy. Each dictionary, at its selected pi_bar, is
   swept over the delta0 grid of the frontiers (delta = delta0^2 Var_Pi); the
   dictionary with the higher mean validation accuracy is kept, and the
   enriched one in case of a tie (difference < TIE).

Writes results/tuning_<dataset>_<constraint>.csv (grid; model = STABLE for
the enriched dictionary, STABLE_noCF without the label-flipped components)
and results/tuning_<dataset>_<constraint>_dictionary.csv (step 2). The
heatmaps are drawn by plots/heatmaps.py.

Usage
-----
    python experiments/tuning.py --dataset adult --constraint DP
"""

import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from methods import DELTA0_GRID, RESULTS, var_pi
from stable import ERMBaseline, SOCPFairnessMixture
from stable.data import ACS_TRAIN_STATE, ACS_VALIDATION_STATES, load, load_acs
from stable.metrics import accuracy, evaluate, targets

warnings.filterwarnings("ignore")

PI_BARS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
DELTAS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
DICTIONARIES = {"STABLE": True, "STABLE_noCF": False}    # model -> add_counterfactual
TIE = 0.005

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=["adult", "compas", "acs"])
p.add_argument("--constraint", required=True, choices=["DP", "EO"])
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
C, seed = args.constraint, args.seed

if args.dataset == "acs":
    Xtr, Gtr, ytr = load_acs(ACS_TRAIN_STATE, max_n=80_000, seed=seed)   # as in acs_pareto.py
    val = [load_acs(s, max_n=30_000, seed=seed) for s in ACS_VALIDATION_STATES]
else:
    X, G, y = load(args.dataset)
    Xtr, Xva, Gtr, Gva, ytr, yva = train_test_split(
        X, G, y, test_size=0.3, random_state=seed, stratify=y)
    val = [(Xva, Gva, yva)]

erm = ERMBaseline().fit(Xtr, Gtr, ytr)
t = targets(erm.predict_proba(Xtr), Gtr, ytr)[0 if C == "DP" else 1]


def scores(model):
    """Mean accuracy, U and deploy gap over the validation set(s). For ACS the
    deploy gap is |Phi(h; pi_state) - t| at each state's own proportion."""
    res = []
    for Xv, Gv, yv in val:
        h = model.predict_proba(Xv)
        r = evaluate(h, Gv, yv, t, C)
        if args.dataset == "acs":
            pos = yv == 1
            phi = float(np.mean(h)) if C == "DP" else float(np.mean(h[pos]))
            r["deploy"] = abs(phi - t)
        res.append((accuracy(h, yv), r["u"], r["deploy"]))
    return np.mean(res, axis=0)


def fit_score(cf, pi_bar, delta):
    try:
        m = SOCPFairnessMixture(pi_bar=pi_bar, delta=delta, t=t, constraint=C,
                                add_counterfactual=cf)
        return scores(m.fit(Xtr, Gtr, ytr, seed=seed))
    except RuntimeError:
        return np.nan, np.nan, np.nan


# -- 1. grid over (dictionary, pi_bar, delta) ---------------------------------
rows = [dict(model="ERM", pi_bar=np.nan, delta=np.nan,
             **dict(zip(["acc", "u", "deploy"], scores(erm))))]
for (model, cf), pi_bar, delta in product(DICTIONARIES.items(), PI_BARS, DELTAS):
    acc, u, dep = fit_score(cf, pi_bar, delta)
    rows.append(dict(model=model, pi_bar=pi_bar, delta=delta, acc=acc, u=u, deploy=dep))
    print(f"  {model:11s} pi_bar={pi_bar:.2f} delta={delta:.0e}  "
          f"acc={acc:.3f} U={u:.3f} deploy={dep:.3f}", flush=True)

RESULTS.mkdir(exist_ok=True)
grid = pd.DataFrame(rows)
grid.to_csv(RESULTS / f"tuning_{args.dataset}_{C}.csv", index=False)

# -- 2. dictionary, on the delta0 grid of the frontiers ----------------------
rows, summary = [], {}
for model, cf in DICTIONARIES.items():
    best = grid[grid.model == model].sort_values("deploy").iloc[0]
    for delta0 in DELTA0_GRID[args.dataset]:
        acc, u, dep = fit_score(cf, best.pi_bar, delta0 ** 2 * var_pi(best.pi_bar))
        rows.append(dict(model=model, pi_bar=best.pi_bar, delta0=delta0,
                         acc=acc, u=u, deploy=dep))
    d = pd.DataFrame(rows)
    summary[model] = (best.pi_bar, best.delta, d[d.model == model].acc.mean())
pd.DataFrame(rows).to_csv(RESULTS / f"tuning_{args.dataset}_{C}_dictionary.csv", index=False)

print()
for model, (pi_bar, delta, acc) in summary.items():
    print(f"{model:11s} best deploy gap at pi_bar={pi_bar} delta={delta:.0e}; "
          f"mean val. accuracy over the delta0 grid = {acc:.4f}")
gain = summary["STABLE_noCF"][2] - summary["STABLE"][2]
chosen = "STABLE_noCF" if gain >= TIE else "STABLE"
print(f"-> dictionary: {'without' if chosen == 'STABLE_noCF' else 'with'} label-flipped "
      f"components (accuracy difference without - with = {gain:+.4f}, tie threshold {TIE})")
