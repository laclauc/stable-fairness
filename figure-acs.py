"""
ACS figures — standalone script, reads from results/acs_geographic_robustness.csv.
Produces:
  - acs_geo_scatter_tradeoff : scatter (Phi vs pi_state) + tradeoff (U vs Acc) côte à côte
  - acs_geo_barplot          : deploy gap per state (inchangé)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_SCATTER  = ["EG", "SOCP", "PostProc", "Shifty"]
MODELS_BARPLOT  = ["EG", "SOCP", "PostProc", "Shifty"]
MODELS_TRADEOFF = ["EG", "SOCP", "PostProc", "Shifty"]

REGISTRY = {
    "ERM":      ("phi_erm",    "dep_erm",    "u_erm",    "acc_erm",    "#d62728", "o", "ERM"),
    "EG":       ("phi_eg",     "dep_eg",     "u_eg",     "acc_eg",     "#ff7f0e", "s", "ERM + DP"),
    "SOCP":     ("phi_socp",   "dep_socp",   "u_socp",   "acc_socp",   "#1f77b4", "*", "STABLE"),
    "PostProc": ("phi_pp",     "dep_pp",     "u_pp",     "acc_pp",     "#2ca02c", "^", "PostProc"),
    "Shifty":   ("phi_shifty", "dep_shifty", "u_shifty", "acc_shifty", "#9467bd", "D", "Shifty"),
}

FS_L, FS_T, FS_K = 18, 15, 16

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

df_raw = pd.read_csv(RES_DIR / "acs_geographic_robustness.csv")

agg_dict = {"pi": ("pi", "mean"), "t": ("t", "mean")}
for key, (phi_col, dep_col, u_col, acc_col, *_) in REGISTRY.items():
    for col in (phi_col, dep_col, u_col, acc_col):
        agg_dict[col]              = (col, "mean")
        agg_dict[f"{col}_std"]     = (col, "std")

df     = df_raw.groupby("state").agg(**agg_dict).reset_index().sort_values("pi")
t_mean = float(df["t"].mean())
print(f"Loaded {len(df_raw)} rows, {df_raw.seed.nunique()} seeds, {len(df)} states.")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

# ---------------------------------------------------------------------------
# Figure 1 — Scatter + Tradeoff côte à côte, légende unique
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ── Scatter : Phi(h; pi_state) vs pi_state ────────────────────────────────────
for key in MODELS_SCATTER:
    phi_col, _, _, _, color, marker, label = REGISTRY[key]
    ax1.scatter(df["pi"], df[phi_col], color=color, marker=marker,
                s=120, zorder=4, label=label)
    if f"{phi_col}_std" in df.columns:
        ax1.errorbar(df["pi"], df[phi_col], yerr=df[f"{phi_col}_std"],
                     fmt="none", color=color, capsize=4, lw=1.5)
    z    = np.polyfit(df["pi"], df[phi_col], 1)
    pi_l = np.linspace(df["pi"].min() - 0.02, df["pi"].max() + 0.02, 100)
    ax1.plot(pi_l, np.poly1d(z)(pi_l), color=color, lw=2, ls="--", alpha=0.5)

label_col = REGISTRY[MODELS_SCATTER[0]][0]
label_color = REGISTRY[MODELS_SCATTER[0]][4]
for _, row in df.iterrows():
    ax1.annotate(row["state"], xy=(row["pi"], row[label_col]),
                 xytext=(5, 5), textcoords="offset points",
                 fontsize=10, color=label_color)

ax1.axhline(t_mean, color="black", lw=1.2, ls=":", alpha=0.7)
ax1.text(1.015, t_mean, r"$t$", va="center", fontsize=FS_L,
         transform=ax1.get_yaxis_transform())
ax1.set_xlabel(r"State proportion of White residents $\pi_{\rm state}$", fontsize=FS_L)
ax1.set_ylabel(r"$\Phi(h;\,\pi_{\rm state})$", fontsize=FS_L)
ax1.tick_params(labelsize=FS_T)
ax1.grid(True, alpha=0.2)

# ── Tradeoff : U(h) vs Accuracy ───────────────────────────────────────────────
for key in MODELS_TRADEOFF:
    _, _, u_col, acc_col, color, marker, label = REGISTRY[key]
    ax2.scatter(df[u_col], df[acc_col], color=color, marker=marker,
                s=130, zorder=4, label=label)
    if f"{u_col}_std" in df.columns and f"{acc_col}_std" in df.columns:
        ax2.errorbar(df[u_col], df[acc_col],
                     xerr=df[f"{u_col}_std"], yerr=df[f"{acc_col}_std"],
                     fmt="none", color=color, capsize=3, lw=1.2, alpha=0.6)
    for _, row in df.iterrows():
        ax2.annotate(row["state"], xy=(row[u_col], row[acc_col]),
                     xytext=(4, 3), textcoords="offset points",
                     fontsize=9, color=color, alpha=0.85)

ax2.set_xlabel(r"$U(h)$ — unfairness gap $\downarrow$", fontsize=FS_L)
ax2.set_ylabel(r"Accuracy $\uparrow$", fontsize=FS_L)
ax2.tick_params(labelsize=FS_T)
ax2.grid(True, alpha=0.2)

# ── Légende unique au-dessus ───────────────────────────────────────────────────
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(MODELS_SCATTER),        # une seule ligne
    fontsize=FS_K,
    framealpha=0.9,
    borderaxespad=0.0,
)
fig.tight_layout()
fig.subplots_adjust(top=0.88)        # espace sous la légende
save_fig(fig, "acs_geo_scatter_tradeoff")

# ---------------------------------------------------------------------------
# Figure 2 — Barplot : deploy gap per state (inchangé)
# ---------------------------------------------------------------------------

n_models = len(MODELS_BARPLOT)
w        = 0.8 / n_models
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df))

for offset, key in zip(offsets, MODELS_BARPLOT):
    _, dep_col, _, _, color, _, label = REGISTRY[key]
    ax.bar(x + offset, df[dep_col], width=w, color=color, label=label, alpha=0.88)
    if f"{dep_col}_std" in df.columns:
        ax.errorbar(x + offset, df[dep_col], yerr=df[f"{dep_col}_std"],
                    fmt="none", color="black", capsize=3, lw=1.2)

ax.set_xticks(x)
ax.set_xticklabels(df["state"].values, fontsize=FS_T)
ax.set_ylabel(r"$|\Phi(h;\,\pi_{\rm state}) - t|$", fontsize=FS_L)
ax.tick_params(labelsize=FS_T)
ax.legend(fontsize=FS_K, loc="best",
          ncol=3 if len(MODELS_BARPLOT) > 2 else 1)
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(bottom=0)
save_fig(fig, "acs_geo_barplot")