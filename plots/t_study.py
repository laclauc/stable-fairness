"""Effect of the target t on STABLE (Adult): accuracy, U(h) and deployment
gap as functions of t, with ERM as reference. Reads results/t_study_adult.csv.

Usage
-----
    python plots/t_study.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from style import COLOR, RESULTS, paper_style, save

paper_style()
df = pd.read_csv(RESULTS / "t_study_adult.csv")

for c in ("DP", "EO"):
    st = df[(df.model == "STABLE") & (df.constraint == c)]
    erm = df[(df.model == "ERM") & (df.constraint == c)]
    t_erm, t_erm_sd = erm.t.mean(), erm.t.std()
    # one point per grid value of t, plus one for t_ERM (aggregated over seeds,
    # since t_ERM differs slightly from seed to seed)
    groups = [(tv, g) for tv, g in st[~st.is_t_erm].groupby("t")]
    groups = sorted(groups + [(t_erm, st[st.is_t_erm])], key=lambda x: x[0])
    ts = np.array([tv for tv, _ in groups])

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.6))
    for ax, (m, name) in zip(axes, [("acc", "Accuracy"), ("u", "Unfairness"),
                                    ("deploy", "Deployment gap")]):
        mu = np.array([g[m].mean() for _, g in groups])
        sd = np.array([g[m].std() for _, g in groups])
        ax.fill_between(ts, mu - sd, mu + sd, color=COLOR["STABLE"], alpha=0.12)
        ax.plot(ts, mu, "--*", color=COLOR["STABLE"], lw=0.8, ms=3)
        e_mu, e_sd = erm[m].mean(), erm[m].std()
        ax.axhline(e_mu, color=COLOR["ERM"], ls="--", lw=0.8)
        ax.axhspan(e_mu - e_sd, e_mu + e_sd, color=COLOR["ERM"], alpha=0.1)
        ax.axvspan(t_erm - t_erm_sd, t_erm + t_erm_sd, color="black", alpha=0.08)
        ax.axvline(t_erm, color="black", ls=":", lw=0.8)
        ax.set_xlabel("$t$")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.2)
    fig.legend([Line2D([], [], color=COLOR["STABLE"], ls="--", marker="*"),
                Line2D([], [], color=COLOR["ERM"], ls="--")], ["STABLE", "ERM"],
               loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, f"t_study_adult_{c}")
