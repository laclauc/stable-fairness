"""Shared plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt

COLORS     = {"ERM": "#d62728", "EG": "#ff7f0e", "SOCP": "#1f77b4",
              "PostProc": "#2ca02c", "Shifty": "#9467bd"}
ALPHA_BAND = 0.13
FS_LABEL   = 18
FS_TICK    = 15
FS_LEGEND  = 15
FS_T       = 18


def _curve_stats(curves):
    """Mean + centered std (removes level shift across seeds, keeps slope variance)."""
    arr = np.array(curves)
    std = (arr - arr.mean(axis=1, keepdims=True)).std(axis=0)
    return arr.mean(axis=0), std


def plot_phi_panel(ax, curves_erm, curves_eg, curves_socp, pi_grid, t_mean,
                   ylabel=True, extra_curves=None):
    """Plot phi curves with confidence bands for one constraint panel.

    Parameters
    ----------
    extra_curves : list of (curves, color, label) tuples, optional
        Additional methods to overlay (e.g. PostProc, Shifty).
    """
    panels = [
        (curves_erm,  COLORS["ERM"],  "ERM"),
        (curves_eg,   COLORS["EG"],   "ERM + Constraint"),
        (curves_socp, COLORS["SOCP"], "STABLE"),
    ]
    if extra_curves is not None:
        panels.extend(extra_curves)

    for curves, color, label in panels:
        mean, std = _curve_stats(curves)
        ax.plot(pi_grid, mean, color=color, lw=4, label=label)
        ax.fill_between(pi_grid, mean - std, mean + std, color=color, alpha=ALPHA_BAND)

    ax.axhline(t_mean, color="black", lw=3.0, ls=":", alpha=0.7)
    ax.text(1.015, t_mean, r"$t$", va="center", fontsize=FS_T,
            transform=ax.get_yaxis_transform())

    ax.set_xlabel(r"Deployment group proportion $\pi'$", fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel(r"$\Phi(h;\,\pi')$", fontsize=FS_LABEL)

    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)

    ax.legend(
        fontsize=FS_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01), 
        framealpha=0.9,
        ncol=3,
        borderaxespad=0.0,
    )

    ax.grid(True, alpha=ALPHA_BAND)


def plot_heatmap(ax, data, cmap, vmin, vmax, delta_labels, pibar_labels):
    """Heatmap for hyperparameter grid search results."""
    n_pibar, n_delta = data.shape
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_pibar):
        for j in range(n_delta):
            val = data[i, j]
            if not np.isnan(val):
                norm  = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = "white" if norm > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    ax.set_xticks(range(n_delta))
    ax.set_xticklabels(delta_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_pibar))
    ax.set_yticklabels(pibar_labels, fontsize=9)
    ax.set_xlabel(r"$\delta$", fontsize=12)
    ax.set_ylabel(r"$\bar\pi$", fontsize=12)


def save_fig(fig, name, fig_dir):
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(fig_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")
