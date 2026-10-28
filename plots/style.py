"""Shared style and data helpers for the figure scripts."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT))

COLOR = {"ERM": "crimson", "STABLE": "tab:blue", "EG": "tab:orange",
         "PostProc": "tab:green", "Shifty": "tab:purple", "DRO": "tab:brown",
         "STABLE_noCF": "tab:cyan", "STABLE_CF": "tab:blue"}
ORDER = ["STABLE", "STABLE_CF", "STABLE_noCF", "EG", "PostProc", "DRO", "Shifty"]
NICE = {"adult": "Adult", "compas": "COMPAS", "acs": "ACS Income"}


def label(method, constraint=None):
    return {"EG": f"ERM + {constraint}" if constraint else "ERM + constraint",
            "DRO": r"STABLE$_{\mathrm{DRO}}$",
            "STABLE_CF": "With label-flipped components",
            "STABLE_noCF": "Without label-flipped components"}.get(method, method)


def paper_style():
    """Fonts sized for panels printed at ~1.35in (NeurIPS text width / 4)."""
    plt.rcParams.update({
        "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42,
    })


def load_runs(spec):
    """Read and merge result files. `spec` is 'path' or 'path1+path2+...':
    rows of the first file are kept as is, later files only contribute their
    non-ERM rows (each run writes its own ERM reference)."""
    paths = [Path(s) for s in spec.split("+")]
    parts = [pd.read_csv(paths[0])]
    for path in paths[1:]:
        d = pd.read_csv(path)
        parts.append(d[d.method != "ERM"])
    df = pd.concat(parts, ignore_index=True)
    if "ok" not in df.columns:              # older result files
        df["ok"] = df["acc"].notna()
    df["ok"] = df["ok"].astype(bool)
    keys =[k for k in ("method", "delta0", "seed", "state") if k in df.columns]
    dup = df[df.method != "ERM"].duplicated(keys).sum()
    if dup:
        raise SystemExit(f"{dup} duplicated rows when merging {spec}")
    return df


def parse_inputs(entries, default_pattern):
    """'dataset:CONSTRAINT=spec' entries -> list of (dataset, constraint, df)."""
    out = []
    for e in entries:
        key, spec = e.split("=", 1) if "=" in e else (e, None)
        ds, c = key.split(":")
        out.append((ds, c, load_runs(spec or str(RESULTS / default_pattern.format(ds=ds, c=c)))))
    return out


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.pdf")


PANEL_W, PANEL_H = 1.33, 1.15            # inches; 4 panels + pad = 5.4in text width
_RECT = [0.25, 0.26, 0.71, 0.70]         # axes box inside each panel


def panel_grid(n_cols, n_rows, handles, col_labels=None, left_pad=0.08, right_pad=0.06):
    """Figure of n_rows x n_cols identical panels, with a shared legend on top
    and optional '(a) ...' labels under the columns. `left_pad` (inches) leaves
    room for the first column's y label, `right_pad` for the last column's
    rightmost tick label. Returns (fig, axes[row][col])."""
    # one legend row if it fits, two otherwise
    ncol = len(handles) if len(handles) <= n_cols + 2 else -(-len(handles) // 2)
    leg_h, lab_h = (0.22 if ncol == len(handles) else 0.34), (0.16 if col_labels else 0.0)
    W, H = left_pad + n_cols * PANEL_W + right_pad, leg_h + n_rows * PANEL_H + lab_h
    fig = plt.figure(figsize=(W, H))

    def x_of(j, frac):                     # panel j, fraction of its width
        return (left_pad + (j + frac) * PANEL_W) / W

    axes = [[fig.add_axes([x_of(j, _RECT[0]),
                           (H - leg_h - (i + 1) * PANEL_H + _RECT[1] * PANEL_H) / H,
                           _RECT[2] * PANEL_W / W, _RECT[3] * PANEL_H / H])
             for j in range(n_cols)] for i in range(n_rows)]
    for j, text in enumerate(col_labels or []):
        fig.text(x_of(j, _RECT[0] + _RECT[2] / 2), lab_h / 2 / H,
                 f"({chr(97 + j)}) {text}", ha="center", va="center")
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1 - leg_h / 2 / H),
               ncol=ncol, frameon=False, handlelength=1.8, columnspacing=1.4)
    return fig, axes
