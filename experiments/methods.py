"""Method factories shared by the experiment scripts.

Every factory returns a fitted object exposing predict_proba(X) -> P(yhat=1|x).
All methods are swept on a common fairness-budget grid delta0:

    STABLE            delta = delta0^2 * Var_Pi(pi),  Var_Pi = (1 - 2 pi_bar)^2 / 12
                      (so that U(h) <~ delta0, Corollary 1)
    ERM + constraint  fairlearn ExponentiatedGradient, difference_bound = eps = delta0
    PostProc          constraint slack eps = delta0              (optional, see README)
    Shifty            epsilon_fairness = delta0                  (optional, see README)
    STABLE_DRO        own grid over its fairness weight lambda (stable/dro.py)
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.reductions import (DemographicParity, ExponentiatedGradient,
                                  TruePositiveRateParity)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from stable import DROMixture, SOCPFairnessMixture  # noqa: E402

# Default deployment range Pi = [pi_bar, 1 - pi_bar], per constraint.
PI_BAR = {"DP": 0.02, "EO": 0.15}

# Values selected on validation data by experiments/tuning.py (paper, Table of
# hyperparameters): pi_bar and whether the dictionary includes the
# label-flipped components. Used by STABLE and STABLE_DRO in pareto.py and
# acs_pareto.py, and for the range of the worst-case accuracy.
SELECTED = {("adult", "DP"): dict(pi_bar=0.02, add_counterfactual=True),
            ("adult", "EO"): dict(pi_bar=0.15, add_counterfactual=True),
            ("compas", "DP"): dict(pi_bar=0.02, add_counterfactual=False),
            ("compas", "EO"): dict(pi_bar=0.02, add_counterfactual=True),
            ("acs", "DP"): dict(pi_bar=0.05, add_counterfactual=True),
            ("acs", "EO"): dict(pi_bar=0.15, add_counterfactual=True)}

# Common fairness-budget grid of the frontiers (STABLE: delta = delta0^2 Var_Pi).
DELTA0_GRID = {"adult": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15],
               "compas": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15],
               "acs": [0.005, 0.01, 0.02, 0.04, 0.08, 0.15]}

# Fairness weight lambda of STABLE_DRO: no correspondence with delta0, so it is
# swept on its own grid, per setting, covering the transition from the ERM
# solution to the fair regime (paper, Appendix E.1).
_LAM_ACS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 0.95, 0.99]
_LAM_COMPAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
LAM_GRID = {("adult", "DP"): [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7],
            ("adult", "EO"): [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
            ("compas", "DP"): _LAM_COMPAS, ("compas", "EO"): _LAM_COMPAS,
            ("acs", "DP"): _LAM_ACS, ("acs", "EO"): _LAM_ACS}

# Optional third-party baselines: not distributed with this repository.
# Put the authors' code on the Python path to enable them (see README).
try:
    from postproc import PostProcessor
except ImportError:
    PostProcessor = None
try:
    from shifty import run_shifty_dp, run_shifty_eo
except ImportError:
    run_shifty_dp = run_shifty_eo = None


def var_pi(pi_bar):
    return (1.0 - 2.0 * pi_bar) ** 2 / 12.0


class _Proba:
    def __init__(self, fn):
        self.predict_proba = fn


def fit_stable(X, G, y, t, delta0, constraint, seed, pi_bar=None, add_counterfactual=True):
    pi_bar = PI_BAR[constraint] if pi_bar is None else pi_bar
    return SOCPFairnessMixture(
        pi_bar=pi_bar, delta=delta0 ** 2 * var_pi(pi_bar), t=t,
        constraint=constraint, add_counterfactual=add_counterfactual,
    ).fit(X, G, y, seed=seed)


def fit_erm_constraint(X, G, y, t, delta0, constraint, seed):
    """fairlearn's reductions approach. The tolerance is the constraint's
    `difference_bound`; TruePositiveRateParity is equal opportunity.
    EG is a randomized classifier: P(yhat = 1 | x) is given by _pmf_predict."""
    cons = (DemographicParity(difference_bound=delta0) if constraint == "DP"
            else TruePositiveRateParity(difference_bound=delta0))
    eg = ExponentiatedGradient(LogisticRegression(C=1.0, max_iter=1000),
                               constraints=cons, eps=delta0)
    eg.fit(X, y, sensitive_features=G)
    return _Proba(lambda Z: eg._pmf_predict(Z)[:, 1])


def fit_postproc(X, G, y, t, delta0, constraint, seed, T=20_000):
    if PostProcessor is None:
        raise ImportError("PostProc is not distributed with this repository (see README)")
    Xa, Xu, _, _, ya, _ = train_test_split(X, G, y, test_size=0.5, random_state=seed)
    kw = dict(base_estimator=LogisticRegression(C=1.0, max_iter=1000).fit(Xa, ya),
              tau_estimator=LogisticRegression(C=1.0, max_iter=1000).fit(X, G),
              p_S=np.array([np.mean(G == 0), np.mean(G == 1)]),
              constraint=constraint, eps=delta0, T=T, compute_sigma=False)
    if constraint == "EO":
        kw["joint_estimator"] = [
            LogisticRegression(C=1.0, max_iter=1000).fit(X, ((G == s) & (y == 1)).astype(int))
            for s in (0, 1)]
        kw["p_Y1"] = float(np.mean(y == 1))
        kw["p_S_Y1"] = np.array([np.mean((G == 0) & (y == 1)), np.mean((G == 1) & (y == 1))])
    post = PostProcessor(**kw)
    post.fit(Xu, seed=seed)
    return post


def fit_shifty(X, G, y, t, delta0, constraint, seed):
    if run_shifty_dp is None:
        raise ImportError("Shifty is not distributed with this repository (see README)")
    fn = run_shifty_dp if constraint == "DP" else run_shifty_eo
    sh = fn(X, G, y, epsilon_fairness=delta0, setting=2, alpha=0.25, seed=seed)
    if not getattr(sh, "accepted", True):
        # Shifty abstains ("No Solution Found"): record no model rather than
        # its unconstrained fallback.
        raise RuntimeError("Shifty: no solution found")
    return sh


FACTORIES = {
    "STABLE": fit_stable,
    "EG": fit_erm_constraint,
    "PostProc": fit_postproc,
    "Shifty": fit_shifty,
    # dictionary ablation: STABLE with / without the label-flipped components,
    # whatever the dictionary selected for the setting
    "STABLE_CF": lambda *a, **k: fit_stable(*a, **{**k, "add_counterfactual": True}),
    "STABLE_noCF": lambda *a, **k: fit_stable(*a, **{**k, "add_counterfactual": False}),
}


class DROSweep:
    """STABLE_DRO for one seed: the dictionary is fitted once, then each
    lambda only solves a small linear program."""

    def __init__(self, X, G, y, t, constraint, seed, rho=0.9, pi_bar=None,
                 add_counterfactual=True):
        pi_bar = PI_BAR[constraint] if pi_bar is None else pi_bar
        self.model = DROMixture(pi_bar=pi_bar, constraint=constraint, rho=rho,
                                add_counterfactual=add_counterfactual
                                ).fit(X, G, y, t, seed=seed)

    def __call__(self, lam):
        return self.model.with_lambda(lam)


def parse_methods(arg, default):
    methods = [m.strip() for m in (arg or default).split(",")]
    unknown = [m for m in methods if m not in FACTORIES and m != "DRO"]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; "
                         f"available: {list(FACTORIES) + ['DRO']}")
    missing = [m for m, ok in [("PostProc", PostProcessor is not None),
                               ("Shifty", run_shifty_dp is not None)]
               if m in methods and not ok]
    if missing:
        raise SystemExit(f"{missing}: third-party code not found on the Python path "
                         "(not distributed with this repository, see README)")
    return methods
