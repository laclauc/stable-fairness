"""Main-text figure: accuracy vs U(h) (top) and vs deployment gap (bottom) on
Adult and COMPAS, DP and EO. Mean over seeds; bands: +/- one std along x.

Usage
-----
    python plots/pareto_main.py
    # merge extra runs, e.g. a separate Shifty run:
    python plots/pareto_main.py --inputs \
        adult:DP=results/pareto_adult_DP.csv+results/pareto_adult_DP_shifty.csv ...
"""

import argparse

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from style import COLOR, NICE, ORDER, label, paper_style, panel_grid, parse_inputs, save

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+",
               default=["adult:DP", "adult:EO", "compas:DP", "compas:EO"])
p.add_argument("--methods", default="STABLE,EG,PostProc,DRO")
p.add_argument("--out", default="pareto_main")
p.add_argument("--labels", default="",
               help="legend overrides, e.g. 'STABLE=With label-flipped,STABLE_noCF=Without'")
args = p.parse_args()

paper_style()
LABELS = dict(kv.split("=", 1) for kv in args.labels.split(",") if kv)
methods = [m for m in ORDER if m in args.methods.split(",")]
data = parse_inputs(args.inputs, "pareto_{ds}_{c}.csv")
present = [m for m in methods if any(m in set(df.method) for _, _, df in data)]

handles = [Line2D([], [], color=COLOR[m], marker="o", lw=1.1, ms=2.2, label=LABELS.get(m, label(m)))
           for m in present]
handles.append(Line2D([], [], color=COLOR["ERM"], marker="*", lw=0, ms=5, label="ERM"))
fig, axes = panel_grid(len(data), 2, handles,
                       [f"{NICE[ds]}, {c}" for ds, c, _ in data])

for j, (ds, c, df) in enumerate(data):
    ok = df[df.ok & (df.method != "ERM")]
    agg = ok.groupby(["method", "delta0"])[["acc", "u", "deploy"]].agg(["mean", "std"])
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()
    erm = df[df.method == "ERM"]
    for i, (x, xlabel) in enumerate([("u", r"$\mathcal{U}(h)$"), ("deploy", "Deploy gap")]):
        ax = axes[i][j]
        for m in present:
            s = agg[agg.method == m].sort_values("delta0")
            if s.empty:
                continue
            b = s.sort_values("acc_mean")      # band drawn along accuracy: never self-intersects
            sd = b[f"{x}_std"].fillna(0)
            ax.fill_betweenx(b.acc_mean, b[f"{x}_mean"] - sd, b[f"{x}_mean"] + sd,
                             color=COLOR[m], alpha=0.12, lw=0)
            ax.plot(s[f"{x}_mean"], s.acc_mean, "o-", color=COLOR[m], lw=1.1, ms=2.2)
        ax.scatter([erm[x].mean()], [erm.acc.mean()], color=COLOR["ERM"], marker="*",
                   s=28, zorder=5, linewidths=0)
        ax.set_xlabel(xlabel + r" $\downarrow$", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"Accuracy $\uparrow$", labelpad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(pad=1.5)

save(fig, args.out)
