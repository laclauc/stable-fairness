"""
Shifty-ttest integration (Giguere et al., ICLR 2022).
Single module for all datasets — pass epsilon_fairness explicitly.

Usage:
    from utils import phi_dp, ...
    from shifty import run_shifty_dp, run_shifty_eo

    # Adult DP (epsilon=0.10 recommended)
    shifty = run_shifty_dp(X_train, G_train, y_train, epsilon_fairness=0.10)

    # COMPAS DP (epsilon=0.05 recommended)
    shifty = run_shifty_dp(X_train, G_train, y_train, epsilon_fairness=0.05)

Import order matters: utils.py must be imported before this module.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy import optimize as sp_opt

# ---------------------------------------------------------------------------
# Locate Shifty code
# ---------------------------------------------------------------------------

SHIFTY_ROOT = Path(__file__).parent / "Fairness-Guarantees-under-Demographic-Shift" / "Python"
assert SHIFTY_ROOT.exists(), (
    f"Shifty folder not found: {SHIFTY_ROOT}\n"
    "Extract the zip next to this script."
)
sys.path.insert(0, str(SHIFTY_ROOT))

# ---------------------------------------------------------------------------
# Import Shifty, preserving caller's utils.py in sys.modules
# ---------------------------------------------------------------------------

_saved = {k: v for k, v in sys.modules.items()
          if k == "utils" or k.startswith("utils.")
          or k == "datasets" or k.startswith("datasets.")}
for k in list(_saved):
    del sys.modules[k]

import utils.experiments.demographic_shift as _ds
import utils.rvs.utils  as _rvs_utils
import utils.rvs.rvs    as _rvs_mod
from utils.constraints import make_constraints as _make_constraints
from utils.rvs         import get_parser        as _get_parser
from datasets.dataset  import ClassificationDataset as _Dataset
from core.base.sc      import SeldonianClassifier   as _SC
from core.optimizers   import bfgs as _bfgs_mod, OPTIMIZERS as _OPTIMIZERS

sys.modules.update(_saved)

# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

# numpy >= 2.0 repr bug
_orig = _ds._compute_population_conditional
_ds._compute_population_conditional = (
    lambda C, D_str, D_u, cm: _orig(C, D_str, [int(d) for d in D_u], cm)
)

# BFGSOptimizer missing theta0 parameter
class _PatchedBFGS(_bfgs_mod.BFGSOptimizer):
    def minimize(self, evalf, n_iters, theta0=None):
        best_theta, best_val = None, np.inf
        for _ in range(self.restarts):
            x0  = 2.0 * np.random.random(self.n_features) - 1.0
            res = sp_opt.minimize(evalf, x0, method="Powell",
                                  options={"maxiter": n_iters})
            if res.x is not None:
                val = evalf(res.x)
                if val < best_val:
                    best_theta, best_val = res.x.copy(), val
        return best_theta, {}

_bfgs_mod.BFGSOptimizer  = _PatchedBFGS
_OPTIMIZERS["bfgs"]      = _PatchedBFGS

# Replace shgo with SLSQP (faster and more stable)
def _fast_oos(f, bounds):
    bounds       = np.array(bounds, dtype=float)
    inner_bounds = bounds[:-1]
    bl, bu       = bounds[-1]
    def _f(x):
        return f(np.concatenate((x, [1.0 - x.sum()])))
    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum() - bl},
        {"type": "ineq", "fun": lambda w: bu - 1.0 + w.sum()},
    ]
    x0  = np.array([(lo + hi) / 2.0 for lo, hi in inner_bounds])
    res = sp_opt.minimize(_f, x0, method="SLSQP", bounds=inner_bounds,
                          constraints=constraints, options={"ftol": 1e-8, "maxiter": 300})
    x = res.x if res.success else x0
    return np.concatenate((x, [1.0 - x.sum()]))

_rvs_utils.optimize_on_simplex = _fast_oos
_rvs_mod.optimize_on_simplex   = _fast_oos

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ShiftyPredictor:
    """Wrapper with predict_proba(X) -> [0,1], compatible with phi_dp/phi_eo."""
    def __init__(self, theta: np.ndarray, accepted: bool):
        self.theta    = theta
        self.accepted = accepted
        if not accepted:
            print("[Shifty] NSF — solution rejected.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(X @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def _make_dataset(X_train, G_train, y_train, seed):
    n      = len(y_train)
    perm   = np.random.default_rng(seed).permutation(n)
    n_cand = n // 2
    X_s    = X_train[perm]
    Y_s    = (2 * y_train[perm] - 1).astype(float)
    S_s    = G_train[perm].astype(int)
    return _Dataset([-1, 1], n_cand, n - n_cand, n_test=0, seed=seed,
                    X=X_s, Y=Y_s, S=S_s, R=S_s)


def _fit(constraint_str, d, setting, alpha, delta, n_iters, seed):
    splits   = d.training_splits()
    demo_arr = splits["S"]
    unique_D = np.unique(demo_arr).astype(int)
    Pr_D     = np.array([(demo_arr == v).mean() for v in unique_D])
    D        = _get_parser(mode="inner").parse("S")
    known    = _ds.get_population_conditionals(d.all_sets(), [constraint_str], D)
    marginals   = _ds.make_intervals(Pr_D, alpha=alpha) if setting == 2 else Pr_D
    robust_loss = (setting == 2)
    model = _SC([constraint_str], [delta], verbose=False, shape_error=True,
                model_type="linear", ci_mode="ttest", cs_scale=2.0,
                robust_loss=robust_loss, demographic_variable=D,
                demographic_variable_values=unique_D, demographic_marginals=marginals,
                known_demographic_terms=known, seed=seed)
    accepted = model.fit(d, n_iters=n_iters, optimizer_name="bfgs")
    return ShiftyPredictor(theta=model.theta, accepted=accepted)


def run_shifty_dp(X_train, G_train, y_train,
                  epsilon_fairness=0.10,
                  setting=2, alpha=0.25,
                  delta=0.05, n_iters=40, seed=42) -> ShiftyPredictor:
    """
    Shifty-ttest with Demographic Parity constraint.
    setting=1: known shift | setting=2: unknown shift (default)
    Recommended epsilon: 0.10 for Adult, 0.05 for COMPAS.
    """
    constraint_str = _make_constraints("DemographicParity", "S", [0, 1], epsilon_fairness)[0]
    print(f"[Shifty DP] {constraint_str}, setting={setting}"
          + (f", alpha={alpha}" if setting == 2 else ""))
    d = _make_dataset(X_train, G_train, y_train, seed)
    return _fit(constraint_str, d, setting, alpha, delta, n_iters, seed)


def run_shifty_eo(X_train, G_train, y_train,
                  epsilon_fairness=0.15,
                  setting=2, alpha=0.25,
                  delta=0.05, n_iters=40, seed=42) -> ShiftyPredictor:
    """
    Shifty-ttest with Equal Opportunity constraint.
    Recommended epsilon: 0.15 (EO is harder than DP).
    """
    constraint_str = _make_constraints("EqualOpportunity", "S", [0, 1], epsilon_fairness)[0]
    print(f"[Shifty EO] {constraint_str}, setting={setting}"
          + (f", alpha={alpha}" if setting == 2 else ""))
    d = _make_dataset(X_train, G_train, y_train, seed)
    return _fit(constraint_str, d, setting, alpha, delta, n_iters, seed)