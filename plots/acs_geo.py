"""Geographic generalization on ACS Income (train CA, 11 held-out states).

(left)  Phi(h; pi_state) against the state's proportion pi_state, with a
        linear fit per method; (right) accuracy against U(h), one point per
        state. Uses the ACS results at one fairness budget (--delta0); STABLE_DRO,
        if requested, at the lambda that minimises its mean U(h).

Usage
-----
    python plots/acs_geo.py --constraint DP
    python plots/acs_geo.py --constraint EO
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from style import COLOR, ORDER, RESULTS, label, load_runs, paper_style, save

p = argparse.ArgumentParser()
p.add_argument("--constraint", required=True, choices=["DP", "EO"])
p.add_argument("--input", default=None, help="file1[+file2...]")
p.add_argument("--delta0", type=float, default=0.01)
p.add_argument("--methods", default="STABLE,EG,PostProc,DRO")
p.add_argument("--out", default=None)
args = p.parse_args()
C = args.constraint

paper_style()
df = load_runs(args.input or str(RESULTS / f"acs_pareto_{C}.csv"))
df = df[df.ok & (df.method != "ERM")]
# methods swept in delta0: the chosen budget; STABLE_DRO: its fairest lambda
dro = df[df.method == "DRO"]
lam = dro.groupby("delta0").u.mean().idxmin() if not dro.empty else np.nan
df = df[np.isclose(df.delta0, np.where(df.method == "DRO", lam, args.delta0))]
methods = [m for m in ORDER if m in args.methods.split(",") and m in set(df.method)]
by_state = (df.groupby(["method", "state"])[["pi", "phi", "u", "acc"]]
            .agg(["mean", "std"]))
by_state.columns = ["_".join(c) for c in by_state.columns]
by_state = by_state.reset_index()
t = df.t.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.1),
                               gridspec_kw=dict(wspace=0.32, left=0.09, right=0.98,
                                                bottom=0.2, top=0.84))
for m in methods:
    s = by_state[by_state.method == m].sort_values("pi_mean")
    ax1.errorbar(s.pi_mean, s.phi_mean, yerr=s.phi_std, fmt="o", color=COLOR[m],
                 ms=2.5, lw=0.6, capsize=1.2, label=label(m, C))
    slope, icpt = np.polyfit(s.pi_mean, s.phi_mean, 1)
    xs = np.linspace(s.pi_mean.min() - 0.02, s.pi_mean.max() + 0.02, 50)
    ax1.plot(xs, slope * xs + icpt, color=COLOR[m], ls="--", lw=0.8, alpha=0.6)
    print(f"{m:10s} slope of pi_state -> Phi: {slope:+.3f}")
    ax2.errorbar(s.u_mean, s.acc_mean, xerr=s.u_std, yerr=s.acc_std, fmt="o",
                 color=COLOR[m], ms=2.5, lw=0.5, capsize=1, alpha=0.85)

stable = by_state[by_state.method == "STABLE"]
for _, r in stable.iterrows():
    ax1.annotate(r.state, (r.pi_mean, r.phi_mean), xytext=(2, 2),
                 textcoords="offset points", fontsize=4.5, color=COLOR["STABLE"])
ax1.axhline(t, color="0.3", ls=":", lw=0.8)
ax1.text(1.01, t, "$t$", transform=ax1.get_yaxis_transform(), va="center")
ax1.set_xlabel(r"State proportion $\pi_{\rm state}$" + (" (White residents)" if C == "DP"
               else r" = $P(G=1\mid Y=1)$"), labelpad=1)
ax1.set_ylabel(r"$\Phi(h;\pi_{\rm state})$", labelpad=1)
ax2.set_xlabel(r"$\mathcal{U}(h)$ $\downarrow$", labelpad=1)
ax2.set_ylabel(r"Accuracy $\uparrow$", labelpad=1)
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.tick_params(pad=1.5)
fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=len(methods),
           frameon=False, bbox_to_anchor=(0.5, 1.0))
save(fig, args.out or f"acs_geo_{C}")
