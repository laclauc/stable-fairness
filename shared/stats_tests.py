"""Shared statistical significance tests (Wilcoxon paired)."""

import numpy as np
from scipy.stats import wilcoxon, shapiro


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def wilcoxon_test(a, b, alternative="less", label=""):
    _, p_norm_a = shapiro(a)
    _, p_norm_b = shapiro(b)
    try:
        _, p = wilcoxon(a, b, alternative=alternative)
    except ValueError:
        p = 1.0
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label:<45} p={p:.4f} {stars}  "
          f"(normality A={p_norm_a:.2f}, B={p_norm_b:.2f})")
    return p, stars


def run_wilcoxon_suite(df, pairs):
    """
    Run the standard test suite.

    pairs: list of (socp_name, baseline_name, constraint)
    e.g. [("SOCP_DP", "EG_DP", "DP"), ("SOCP_EO", "EG_EO", "EO")]
    """
    def get(model, constraint, metric):
        mask = (df["model"] == model) & (df["constraint"] == constraint)
        return df[mask].sort_values("seed")[metric].values

    for socp, baseline, c in pairs:
        section(f"{c} — {socp} vs {baseline}")
        su = get(socp,     c, "u");      bu = get(baseline, c, "u")
        sd = get(socp,     c, "deploy"); bd = get(baseline, c, "deploy")
        sa = get(socp,     c, "acc");    ba = get(baseline, c, "acc")
        ea = get("ERM",    c, "acc")

        print(f"\n  Means:")
        print(f"    {socp:<12} U={su.mean():.3f}±{su.std():.3f}  "
              f"deploy={sd.mean():.3f}±{sd.std():.3f}  acc={sa.mean():.3f}±{sa.std():.3f}")
        print(f"    {baseline:<12} U={bu.mean():.3f}±{bu.std():.3f}  "
              f"deploy={bd.mean():.3f}±{bd.std():.3f}  acc={ba.mean():.3f}±{ba.std():.3f}")
        print(f"\n  Wilcoxon tests (H1: SOCP < baseline):")
        wilcoxon_test(su, bu, "less",    f"U(h):      {socp} < {baseline}")
        wilcoxon_test(sd, bd, "less",    f"Deploy:    {socp} < {baseline}")
        wilcoxon_test(sa, ea, "less",    f"Accuracy:  {socp} < ERM (fairness cost)")

    print(f"\n{'='*70}")
    print("  *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
    print(f"  N = {len(su)} seeds — Wilcoxon signed-rank paired by seed")
    print(f"{'='*70}")
