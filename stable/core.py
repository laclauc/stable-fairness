"""
STABLE: fairness as distributional stability (Algorithm 1).

A convex mixture of predictors is learned by minimising the (linearised)
empirical risk subject to the SOCP constraint

    (1/m) * sum_j (Phi(h; pi'_j) - t)^2 <= delta,

where Phi(h; pi') is the prediction rate of h under the perturbation of the
data distribution indexed by pi'. Two perturbation families are supported:

    DP  (Perturbation 1): the group proportion P(G = 1) is set to pi'.
    EO  (Perturbation 2): P(G = 1 | Y = 1) is set to pi'.

Both are implemented through importance weights, so no resampling is needed.
"""

import numpy as np
import cvxpy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

def compute_weights_dp(G: np.ndarray, pi: float, p1: float) -> np.ndarray:
    """Perturbation 1: w_pi(g) = pi / p1 if g = 1, (1 - pi) / p0 if g = 0."""
    return np.where(G == 1, pi / p1, (1.0 - pi) / (1.0 - p1))


def compute_weights_eo(G: np.ndarray, Y: np.ndarray, pi: float,
                       p11: float, p01: float) -> np.ndarray:
    """Perturbation 2: reweight G within Y = 1, leave Y = 0 unchanged.

    w_pi(g, y) = pi / p11 if (g, y) = (1, 1), (1 - pi) / p01 if (g, y) = (0, 1),
                 1 if y = 0.
    """
    w = np.ones(len(G))
    w[(G == 1) & (Y == 1)] = pi / p11
    w[(G == 0) & (Y == 1)] = (1.0 - pi) / p01
    return w


# ---------------------------------------------------------------------------
# Prediction-rate functionals and unfairness gaps
# ---------------------------------------------------------------------------

def phi_dp(h: np.ndarray, G: np.ndarray, pi: float, p1: float) -> float:
    """Phi_DP(h; pi) = E_pi[h(X)]. `p1` is P(G = 1) on the evaluation set."""
    return float(np.mean(compute_weights_dp(G, pi, p1) * h))


def phi_eo(h: np.ndarray, G: np.ndarray, Y: np.ndarray, pi: float,
           p11: float, p01: float) -> float:
    """Phi_EO(h; pi) = E_pi[h(X) | Y = 1]. `p11`, `p01` from the evaluation set."""
    pos = Y == 1
    if not pos.any():
        return 0.0
    w = compute_weights_eo(G, Y, pi, p11, p01)[pos]
    return float(np.sum(w * h[pos]) / np.sum(w))


def unfairness_gap_dp(h: np.ndarray, G: np.ndarray) -> float:
    """U_DP(h) = |E[h | G = 1] - E[h | G = 0]|."""
    return abs(float(np.mean(h[G == 1])) - float(np.mean(h[G == 0])))


def unfairness_gap_eo(h: np.ndarray, G: np.ndarray, Y: np.ndarray) -> float:
    """U_EO(h) = |E[h | G = 1, Y = 1] - E[h | G = 0, Y = 1]|."""
    pos = Y == 1
    if not pos.any():
        return 0.0
    return abs(float(np.mean(h[pos & (G == 1)])) - float(np.mean(h[pos & (G == 0)])))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """h(x) = val. Always added to the dictionary: its slope is zero, so the
    SOCP is feasible for any target t in [0, 1]."""

    def __init__(self, val: float):
        self.val = val

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.val)


def _logistic(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs")


class ERMBaseline:
    """Unconstrained l2-regularised logistic regression."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.clf_ = None

    def fit(self, X, G, y):
        self.clf_ = _logistic(self.C).fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf_.predict_proba(X)[:, 1]


class SOCPFairnessMixture:
    """STABLE (Algorithm 1).

    Parameters
    ----------
    K : number of weighted-ERM components, trained at K perturbation
        parameters drawn (stratified) from Pi = U[pi_bar, 1 - pi_bar].
    pi_bar : lower bound of the deployment range Pi.
    m : number of perturbation points at which the constraint is evaluated.
    delta : SOCP threshold.
    t : target prediction rate.
    C : inverse l2-regularisation of the logistic components.
    constraint : "DP" or "EO".
    add_counterfactual : enrich the dictionary with label-flipped components
        (Appendix): for each beta on a uniform grid of `K_counterfactual`
        interior points of (0, 1) and each group g, a logistic regression
        trained on labels flipped independently with probability beta within
        group g. This adds 2 * K_counterfactual components.
    K_counterfactual : number of flip probabilities beta.

    The two constant predictors h = 0 and h = 1 are always included.
    """

    def __init__(self, K: int = 30, pi_bar: float = 0.02, m: int = 60,
                 delta: float = 1e-5, t: float = 0.5, C: float = 1.0,
                 constraint: str = "DP", add_counterfactual: bool = True,
                 K_counterfactual: int = 10):
        self.K = K
        self.pi_bar = pi_bar
        self.m = m
        self.delta = delta
        self.t = t
        self.C = C
        self.constraint = constraint.upper()
        if self.constraint not in ("DP", "EO"):
            raise ValueError("constraint must be 'DP' or 'EO'")
        self.add_counterfactual = add_counterfactual
        self.K_counterfactual = K_counterfactual

        self.alpha_ = None
        self.classifiers_ = []
        self.pi_train_ = None
        self.p1_ = self.p11_ = self.p01_ = None

    # -- perturbations -------------------------------------------------------

    def _stratified_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        edges = np.linspace(self.pi_bar, 1.0 - self.pi_bar, n + 1)
        return np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])

    def _weights(self, G, Y, pi):
        if self.constraint == "DP":
            return compute_weights_dp(G, pi, self.p1_)
        return compute_weights_eo(G, Y, pi, self.p11_, self.p01_)

    def _phi(self, h, G, Y, pi):
        if self.constraint == "DP":
            return phi_dp(h, G, pi, self.p1_)
        return phi_eo(h, G, Y, pi, self.p11_, self.p01_)

    # -- dictionary ----------------------------------------------------------

    @staticmethod
    def _predict(clf, X) -> np.ndarray:
        if isinstance(clf, ConstantPredictor):
            return clf.predict_proba(X)
        return clf.predict_proba(X)[:, 1]

    def component_predictions(self, X: np.ndarray) -> np.ndarray:
        """(n, K_total) matrix of the dictionary's predictions on X."""
        return np.column_stack([self._predict(c, X) for c in self.classifiers_])

    def _fit_dictionary(self, X, G, y, rng):
        self.classifiers_ = []
        for pi_k in self.pi_train_:
            w = self._weights(G, y, pi_k)
            self.classifiers_.append(
                _logistic(self.C).fit(X, y, sample_weight=w / w.mean()))
        if self.add_counterfactual and self.K_counterfactual > 0:
            betas = np.linspace(0.0, 1.0, self.K_counterfactual + 2)[1:-1]
            for beta in betas:
                for grp in (1, 0):
                    y_cf = y.copy()
                    in_grp = G == grp
                    flip = rng.random(in_grp.sum()) < beta
                    y_cf[in_grp] = np.where(flip, 1 - y[in_grp], y[in_grp])
                    self.classifiers_.append(_logistic(self.C).fit(X, y_cf))
        self.classifiers_ += [ConstantPredictor(0.0), ConstantPredictor(1.0)]

    def constraint_matrix(self, X, G, y, pi_eval) -> np.ndarray:
        """A[j, k] = Phi(eta_k; pi'_j) on (X, G, y)."""
        P = self.component_predictions(X)
        return np.array([[self._phi(P[:, k], G, y, pi) for k in range(P.shape[1])]
                         for pi in pi_eval])

    def risk_vector(self, X, y) -> np.ndarray:
        """Empirical log-loss of each component (linearised mixture risk)."""
        P = np.clip(self.component_predictions(X), 1e-15, 1 - 1e-15)
        return np.array([log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])
                         for p in P.T])

    # -- SOCP ----------------------------------------------------------------

    def _solve_socp(self, A, r):
        m, K_total = A.shape
        alpha = cp.Variable(K_total)
        prob = cp.Problem(
            cp.Minimize(r @ alpha),
            [alpha >= 0, cp.sum(alpha) == 1,
             cp.norm(A @ alpha - self.t * np.ones(m), 2) <= np.sqrt(m * self.delta)])
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"SOCP did not converge (status: {prob.status}).")
        return np.asarray(alpha.value)

    # -- public API ----------------------------------------------------------

    def fit(self, X, G, y, seed: int = 0):
        rng = np.random.default_rng(seed)
        y = np.asarray(y).astype(int)
        self.p1_ = float(np.mean(G == 1))
        if self.constraint == "EO":
            self.p11_ = float(np.mean(G[y == 1] == 1))
            self.p01_ = 1.0 - self.p11_

        self.pi_train_ = self._stratified_sample(self.K, rng)
        self._fit_dictionary(X, G, y, rng)

        pi_eval = self._stratified_sample(self.m, rng)
        A = self.constraint_matrix(X, G, y, pi_eval)
        r = self.risk_vector(X, y)
        self.alpha_ = self._solve_socp(A, r)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        return np.clip(self.component_predictions(X) @ self.alpha_, 0.0, 1.0)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
