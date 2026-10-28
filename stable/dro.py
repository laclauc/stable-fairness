"""STABLE_DRO: ablation of STABLE with an SA-DRO-style objective.

Same dictionary, same perturbation family and same deployment range as
STABLE; only the objective changes. Following SA-DRO (Lei et al., UAI 2024),
the mixture weights solve the linear program

    min_{alpha in simplex}  max_j [ (1 - lam) (R alpha)_j - rho (pi_j - p1)^2 ]
                            + lam |gamma . alpha|,

where R[j, k] is the importance-weighted log-loss of component k at pi_j and
gamma[k] its signed group gap (linear in alpha, hence in closed form).
lam in [0, 1) is the fairness weight; rho is fixed.

The dictionary, R and gamma do not depend on lam: fit once with
`DROMixture(...).fit(...)`, then call `.with_lambda(lam)` for each lam.
"""

import numpy as np
import cvxpy as cp

from .core import SOCPFairnessMixture, compute_weights_dp, compute_weights_eo


def _logloss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class _Fixed:
    """A mixture with fixed weights over a fitted dictionary."""

    def __init__(self, dictionary, alpha):
        self.dictionary, self.alpha = dictionary, alpha

    def predict_proba(self, X):
        return self.dictionary.component_predictions(X) @ self.alpha


class DROMixture:
    def __init__(self, pi_bar=0.02, m=60, constraint="DP", rho=0.9, **dict_kw):
        self.pi_bar, self.m, self.constraint, self.rho = pi_bar, m, constraint, rho
        self.dict_kw = dict_kw

    def fit(self, X, G, y, t, seed=0):
        # Reuse STABLE's dictionary (delta = 1 makes the SOCP trivially
        # feasible; its alpha is discarded).
        self.dictionary_ = SOCPFairnessMixture(
            pi_bar=self.pi_bar, m=self.m, delta=1.0, t=t,
            constraint=self.constraint, **self.dict_kw).fit(X, G, y, seed=seed)
        P = self.dictionary_.component_predictions(X)
        L = np.column_stack([_logloss(P[:, k], y) for k in range(P.shape[1])])

        self.p1_ = float(np.mean(G == 1))
        p11 = float(np.mean(G[y == 1] == 1))
        self.pi_grid_ = np.linspace(self.pi_bar, 1 - self.pi_bar, self.m)
        self.R_ = np.empty((self.m, P.shape[1]))
        for j, pi in enumerate(self.pi_grid_):
            w = (compute_weights_dp(G, pi, self.p1_) if self.constraint == "DP"
                 else compute_weights_eo(G, y, pi, p11, 1.0 - p11))
            self.R_[j] = (w[:, None] * L).mean(axis=0)

        if self.constraint == "DP":
            m1, m0 = G == 1, G == 0
        else:
            m1, m0 = (G == 1) & (y == 1), (G == 0) & (y == 1)
        self.gamma_ = P[m1].mean(axis=0) - P[m0].mean(axis=0)
        return self

    def with_lambda(self, lam):
        K = self.R_.shape[1]
        a, tt = cp.Variable(K, nonneg=True), cp.Variable()
        pen = self.rho * (self.pi_grid_ - self.p1_) ** 2
        cp.Problem(cp.Minimize(tt + lam * cp.abs(self.gamma_ @ a)),
                   [cp.sum(a) == 1, tt >= (1 - lam) * (self.R_ @ a) - pen]
                   ).solve(solver=cp.CLARABEL)
        if a.value is None:
            raise RuntimeError("DRO linear program failed")
        alpha = np.clip(a.value, 0, None)
        return _Fixed(self.dictionary_, alpha / alpha.sum())
