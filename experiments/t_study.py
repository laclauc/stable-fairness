"""Effect of the target t on STABLE (Adult, DP and EO).

Sweeps t over a grid of 25 values in (0.05, 0.95), plus t_ERM, with the
hyperparameters selected at t_ERM. Writes results/t_study_adult.csv; the
figure is drawn by plots/t_study.py.

Usage
-----
    python experiments/t_study.py --seeds 10
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from methods import RESULTS
from stable import ERMBaseline, SOCPFairnessMixture
from stable.data import load
from stable.metrics import evaluate, targets

warnings.filterwarnings("ignore")

T_GRID = np.linspace(0.05, 0.95, 25)
PARAMS = {"DP": dict(pi_bar=0.02, delta=1e-5), "EO": dict(pi_bar=0.15, delta=1e-5)}

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=10)
args = p.parse_args()

X, G, y = load("adult")
rows = []
for seed in range(args.seeds):
    Xtr, Xte, Gtr, Gte, ytr, yte = train_test_split(X, G, y, test_size=0.3,
                                                    random_state=seed)
    erm = ERMBaseline().fit(Xtr, Gtr, ytr)
    t_erm = dict(zip(["DP", "EO"], targets(erm.predict_proba(Xtr), Gtr, ytr)))
    for c in ("DP", "EO"):
        r = evaluate(erm.predict_proba(Xte), Gte, yte, t_erm[c], c)
        rows.append(dict(model="ERM", constraint=c, seed=seed, t=t_erm[c],
                         is_t_erm=True, acc=r["acc"], u=r["u"], deploy=r["deploy"]))
        for t in sorted(set(np.round(T_GRID, 8)) | {t_erm[c]}):
            try:
                m = SOCPFairnessMixture(**PARAMS[c], t=t, constraint=c)
                r = evaluate(m.fit(Xtr, Gtr, ytr, seed=seed).predict_proba(Xte),
                             Gte, yte, t, c)
            except RuntimeError:
                r = dict(acc=np.nan, u=np.nan, deploy=np.nan)
            rows.append(dict(model="STABLE", constraint=c, seed=seed, t=t,
                             is_t_erm=(t == t_erm[c]), acc=r["acc"], u=r["u"],
                             deploy=r["deploy"]))
    print(f"seed {seed} done", flush=True)

out = RESULTS / "t_study_adult.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"-> {out}")
