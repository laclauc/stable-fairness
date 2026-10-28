"""Hyperparameter sensitivity: accuracy, U(h) and deployment gap of STABLE
over the (pi_bar, delta) grid of experiments/tuning.py.

Usage
-----
    python plots/heatmaps.py --dataset adult --constraint DP
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import RESULTS, paper_style, save

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--constraint", required=True, choices=["DP", "EO"])
p.add_argument("--annotate", action="store_true", help="write the value in each cell")
p.add_argument("--model", default="STABLE", choices=["STABLE", "STABLE_noCF"],
               help="dictionary with (STABLE) or without (STABLE_noCF) label-flipped components")
args = p.parse_args()

paper_style()
df = pd.read_csv(RESULTS / f"tuning_{args.dataset}_{args.constraint}.csv")
grid = df[df.model == args.model]
erm = df[df.model == "ERM"].iloc[0]
pi_bars, deltas = sorted(grid.pi_bar.unique()), sorted(grid.delta.unique())

PANELS = [("acc", "Greens"), ("u", "Blues_r"), ("deploy", "Reds_r")]   # no titles: named in the caption

for metric, cmap in PANELS:
    if grid[metric].isna().all():          # e.g. U is not used for ACS
        continue
    M = (grid.pivot(index="pi_bar", columns="delta", values=metric)
         .reindex(index=pi_bars, columns=deltas).values)
    lo, hi = (np.nanmin(M), erm[metric]) if metric == "acc" else (0.0, max(np.nanmax(M), erm[metric]))
    fig, ax = plt.subplots(figsize=(1.8, 1.6))      # printed at ~0.33 text width
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=4.5)
    for i in range(M.shape[0] if args.annotate else 0):
        for j in range(M.shape[1]):
            v = M[i, j]
            r, g, b, _ = im.cmap(im.norm(v)) if not np.isnan(v) else (1, 1, 1, 1)
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}".lstrip("0"), ha="center",
                    va="center", fontsize=3.6, color="white" if dark else "black")
    ax.set_xticks(range(len(deltas)), [f"{d:.0e}" for d in deltas], rotation=45, ha="right",
                  fontsize=4.5)
    ax.set_yticks(range(len(pi_bars)), [str(v) for v in pi_bars], fontsize=4.5)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\bar\pi$")
    fig.tight_layout()
    suffix = "" if args.model == "STABLE" else "_noCF"
    save(fig, f"{args.dataset}_heatmap_{args.constraint}_{metric}{suffix}")
