"""Prediction rate under demographic shift, pi' -> Phi(h; pi').

Phi is linear in pi', so each (method, seed) curve is the segment between
Phi(h; 0) and Phi(h; 1) (columns phi0, phi1 of the Pareto results). Lines:
mean over seeds; bands: +/- one std of the curve; dashed: the target t.

Operating point: the tightest budget of the common grid (--delta0) for the
methods swept in delta0; STABLE_DRO at the lambda that minimises its mean U(h).

Usage
-----
    python plots/shift_curves.py
"""

import argparse

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from style import COLOR, NICE, ORDER, label, paper_style, panel_grid, parse_inputs, save

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+",
               default=["adult:DP", "adult:EO", "compas:DP", "compas:EO"])
p.add_argument("--delta0", type=float, default=0.005)
p.add_argument("--methods", default="STABLE,EG,PostProc,DRO")
p.add_argument("--out", default="shift_curves")
args = p.parse_args()

paper_style()
methods = ["ERM"] + [m for m in ORDER if m in args.methods.split(",")]
data = parse_inputs(args.inputs, "pareto_{ds}_{c}.csv")
PI = np.linspace(0, 1, 51)


def operating_point(df, m):
    d = df[(df.method == m) & df.ok]
    if m == "DRO" and not d.empty:
        lam = d.groupby("delta0").u.mean().idxmin()
        print(f"  STABLE_DRO at lambda = {lam}")
        return d[np.isclose(d.delta0, lam)]
    return d if m == "ERM" else d[np.isclose(d.delta0, args.delta0)]


present = [m for m in methods if any(m in set(df.method) for _, _, df in data)]
handles = [Line2D([], [], color=COLOR[m], lw=1.1, label=label(m)) for m in present]
handles.append(Line2D([], [], color="0.3", ls="--", lw=0.7, label="target $t$"))
fig, axes = panel_grid(len(data), 1, handles, [f"{NICE[ds]}, {c}" for ds, c, _ in data])

for j, (ds, c, df) in enumerate(data):
    print(f"{ds}/{c}:")
    ax = axes[0][j]
    ax.axhline(df.t.mean(), color="0.3", ls="--", lw=0.7, zorder=1)
    for m in present:
        d = operating_point(df, m)
        if d.empty:
            continue
        curves = np.outer(1 - PI, d.phi0.values) + np.outer(PI, d.phi1.values)
        mu, sd = curves.mean(axis=1), curves.std(axis=1)
        ax.fill_between(PI, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
        ax.plot(PI, mu, color=COLOR[m], lw=1.1)
    ax.set_xlabel(r"$\pi'$", labelpad=1)
    if j == 0:
        ax.set_ylabel(r"$\Phi(h;\pi')$", labelpad=1)
    ax.set_xticks([0, 0.5, 1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.tick_params(pad=1.5)

save(fig, args.out)
