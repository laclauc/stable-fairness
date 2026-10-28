"""LaTeX rows of the appendix tables, from the results of the experiments.

  runtime   wall-clock time of one fit: median [interquartile range] over the
            budgets and seeds; for STABLE_DRO, dictionary and one lambda (the
            first lambda of each seed carries the dictionary fit).
  acs       ACS Income at a fixed budget (delta0, default 0.01; STABLE_DRO at
            the lambda that minimises its mean U): mean accuracy and mean U(h)
            over the held-out states, worst-state deployment gap, |slope| of
            pi_state -> Phi; mean +/- std over seeds.

Usage
-----
    python plots/tables.py runtime
    python plots/tables.py acs
    # merge extra runs, e.g. PostProc / Shifty run separately:
    python plots/tables.py acs --input DP=results/acs_pareto_DP.csv+results/acs_pareto_DP_extra.csv
"""

import argparse

import numpy as np
import pandas as pd

from style import FIGURES, RESULTS, load_runs

METHODS = ["EG", "STABLE", "DRO", "PostProc", "Shifty"]
NAMES = {"ERM": "ERM", "STABLE": "STABLE", "EG": "ERM + constraint",
         "PostProc": "PostProc", "DRO": r"STABLE$_{\text{DRO}}$", "Shifty": "Shifty"}

p = argparse.ArgumentParser()
p.add_argument("table", choices=["runtime", "acs"])
p.add_argument("--input", nargs="*", default=[],
               help="overrides, e.g. adult:DP=a.csv+b.csv (runtime) or DP=a.csv (acs)")
p.add_argument("--delta0", type=float, default=0.01)
args = p.parse_args()
OVERRIDE = dict(e.split("=", 1) for e in args.input)


def decimals(v):
    """3 significant digits for the median; the quartiles use the same decimals."""
    return 2 if v < 10 else (1 if v < 100 else 0)


def runtime_cell(df, m):
    d = df[(df.method == m) & df.ok].drop_duplicates(["method", "delta0", "seed"])
    if m == "DRO":                                   # dictionary + first lambda
        d = d.sort_values("delta0").groupby("seed").head(1)
    if d.empty or "time" not in d:
        return "--"
    q1, med, q3 = np.percentile(d.time, [25, 50, 75])
    k = decimals(med)
    return f"${med:.{k}f}$ {{\\scriptsize[{q1:.{k}f}, {q3:.{k}f}]}}"


def runtime():
    lines = []
    for ds, pattern in [("Adult", "pareto_adult_{c}.csv"), ("COMPAS", "pareto_compas_{c}.csv"),
                        ("ACS", "acs_pareto_{c}.csv")]:
        dfs = {c: load_runs(OVERRIDE.get(f"{ds.lower()}:{c}", str(RESULTS / pattern.format(c=c))))
               for c in ("DP", "EO")}
        cells = [runtime_cell(dfs[c], m) for m in METHODS for c in ("DP", "EO")]
        lines.append(f"{ds} & " + " & ".join(cells) + r" \\")
    return lines


def per_run(df):
    """Collapse the held-out states of each (method, budget, seed) run."""
    def summary(g):
        return pd.Series(dict(acc=g.acc.mean(), u=g.u.mean(), dep=g.deploy.max(),
                              slope=abs(np.polyfit(g.pi, g.phi, 1)[0])))
    return (df[df.ok].groupby(["method", "delta0", "seed"], dropna=False)
            .apply(summary).reset_index())


def acs():
    stats = {}
    for c in ("DP", "EO"):
        r = per_run(load_runs(OVERRIDE.get(c, str(RESULTS / f"acs_pareto_{c}.csv"))))
        dro = r[r.method == "DRO"]
        lam = dro.groupby("delta0").u.mean().idxmin() if not dro.empty else np.nan
        keep = ((r.method == "ERM") | ((r.method == "DRO") & np.isclose(r.delta0, lam))
                | (~r.method.isin(["ERM", "DRO"]) & np.isclose(r.delta0, args.delta0)))
        stats[c] = r[keep].groupby("method")[["acc", "u", "dep", "slope"]].agg(["mean", "std"])
        print(f"% {c}: STABLE_DRO at lambda = {lam}")
    lines = []
    for m in ["ERM", "STABLE", "EG", "PostProc", "DRO", "Shifty"]:
        cells = []
        for c in ("DP", "EO"):
            s = stats[c]
            for k in ("acc", "u", "dep", "slope"):
                if m not in s.index:
                    cells.append("--")
                    continue
                mu, sd = s.loc[m, (k, "mean")], s.loc[m, (k, "std")]
                sd_txt = f"{sd:.3f}".lstrip("0")             # 0.005 -> .005
                cells.append(f"${mu:.3f}${{\\scriptsize$\\pm{sd_txt}$}}")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    return lines


lines = runtime() if args.table == "runtime" else acs()
FIGURES.mkdir(exist_ok=True)
out = FIGURES / f"table_{args.table}.tex"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"% -> {out}")
