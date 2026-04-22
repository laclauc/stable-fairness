"""
Fairness via Distributional Stability - SOCP Mixture Approach
Implementation of Algorithm 1 from the paper.

Supports:
  - Tasks       : "classification" (logistic/MLP) or "regression" (Ridge)
  - Constraints : "DP" (Demographic Parity) or "EO" (Equal Opportunity, classification only)

Key addition: h≡0 and h≡1 are always included in the mixture family,
guaranteeing feasibility of the SOCP for any t in [0,1].
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error
import cvxpy as cp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constant predictor (guarantee feasibility for any t)
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """
    Always predicts a fixed value val in [0,1].
    gamma(h≡val) = 0 for any constraint -> convex hull of {gamma_k} contains 0.
    Works for both classification and regression.
    """
    def __init__(self, val: float):
        self.val = val

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns val for all samples (unified interface)."""
        return np.full(len(X), self.val)


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

def compute_weights_dp(G: np.ndarray, pi: float, p1: float) -> np.ndarray:
    """
    Perturbation 1 (DP): resample G globally.
    w_pi(g) = pi/p1 if g=1,  (1-pi)/p0 if g=0
    """
    p0 = 1.0 - p1
    return np.where(G == 1, pi / p1, (1.0 - pi) / p0)


def compute_weights_eo(
    G: np.ndarray, Y: np.ndarray, pi: float, p11: float, p01: float
) -> np.ndarray:
    """
    Perturbation 2 (EO, classification only): resample G within Y=1.
    w_pi(g, y) = pi/p11 if (g=1,y=1), (1-pi)/p01 if (g=0,y=1), 1 if y=0
    """
    w = np.ones(len(G))
    w[(G == 1) & (Y == 1)] = pi / p11
    w[(G == 0) & (Y == 1)] = (1.0 - pi) / p01
    return w


# ---------------------------------------------------------------------------
# Prediction rate functionals
# ---------------------------------------------------------------------------

def phi_dp(h_pred: np.ndarray, G: np.ndarray, pi: float, p1: float) -> float:
    """
    Phi_DP(h; pi) = E_pi[h(X)].
    Works for both classification (h in {0,1} or [0,1]) and regression (h in [0,1]).
    p1 must be from the EVALUATION set.
    """
    w = compute_weights_dp(G, pi, p1)
    return float(np.mean(w * h_pred))


def phi_eo(
    h_pred: np.ndarray,
    G: np.ndarray,
    Y: np.ndarray,
    pi: float,
    p11: float,
    p01: float,
) -> float:
    """
    Phi_EO(h; pi) = E_pi[h(X) | Y=1] (classification only).
    Normalized by E[w|Y=1] to account for non-unit weights on Y=1.
    p11, p01 must be from the EVALUATION set.
    """
    w    = compute_weights_eo(G, Y, pi, p11, p01)
    mask = (Y == 1)
    if mask.sum() == 0:
        return 0.0
    num = float(np.mean(w[mask] * h_pred[mask]))
    den = float(np.mean(w[mask]))
    return num / den if den > 0 else 0.0


def unfairness_gap_dp(h_pred: np.ndarray, G: np.ndarray) -> float:
    """U_DP(h) = |E[h(X)|G=1] - E[h(X)|G=0]|"""
    r0 = float(np.mean(h_pred[G == 0]))
    r1 = float(np.mean(h_pred[G == 1]))
    return abs(r1 - r0)


def unfairness_gap_eo(h_pred: np.ndarray, G: np.ndarray, Y: np.ndarray) -> float:
    """U_EO(h) = |E[h(X)|G=1,Y=1] - E[h(X)|G=0,Y=1]| (classification only)"""
    mask1 = (Y == 1)
    if mask1.sum() == 0:
        return 0.0
    q0 = float(np.mean(h_pred[(G == 0) & mask1]))
    q1 = float(np.mean(h_pred[(G == 1) & mask1]))
    return abs(q1 - q0)


# ---------------------------------------------------------------------------
# Algorithm 1 - SOCP Mixture
# ---------------------------------------------------------------------------

class SOCPFairnessMixture:
    """
    Convex mixture of predictors minimising risk subject to a
    variance-based fairness constraint (SOCP).

    Parameters
    ----------
    K                 : number of weighted-ERM components
    pi_bar            : lower bound of Pi = U[pi_bar, 1-pi_bar]
    m                 : number of constraint evaluation points
    delta             : SOCP constraint threshold
    t                 : target prediction rate
    C                 : regularisation (logistic) or alpha (Ridge)
    task              : "classification" or "regression"
    model             : "logistic"/"ridge" or "mlp"
    mlp_hidden        : hidden layer sizes for MLP
    constraint        : "DP" or "EO" (EO only for classification)
    add_counterfactual: add interpolated counterfactual components
    K_counterfactual  : number of interpolation steps per group
    """

    def __init__(
        self,
        K: int = 20,
        pi_bar: float = 0.1,
        m: int = 50,
        delta: float = 0.01,
        t: float = 0.5,
        C: float = 1.0,
        task: str = "classification",
        model: str = "logistic",
        mlp_hidden: tuple = (64, 32),
        constraint: str = "DP",
        add_counterfactual: bool = False,
        K_counterfactual: int = 10,
    ):
        self.K                  = K
        self.pi_bar             = pi_bar
        self.m                  = m
        self.delta              = delta
        self.t                  = t
        self.C                  = C
        self.task               = task.lower()
        self.model              = model.lower()
        self.mlp_hidden         = mlp_hidden
        self.constraint         = constraint.upper()
        self.add_counterfactual = add_counterfactual
        self.K_counterfactual   = K_counterfactual

        if self.constraint == "EO" and self.task == "regression":
            raise ValueError("EO constraint is only supported for classification tasks.")

        self.alpha_: np.ndarray | None = None
        self.classifiers_: list = []
        self.pi_train_: np.ndarray | None = None
        self.p1_:  float | None = None
        self.p11_: float | None = None
        self.p01_: float | None = None

    # ------------------------------------------------------------------
    # Stratified sampling from Pi = U[pi_bar, 1-pi_bar]
    # ------------------------------------------------------------------

    def _stratified_sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.pi_bar, 1.0 - self.pi_bar
        edges = np.linspace(lo, hi, n_samples + 1)
        return np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n_samples)])

    # ------------------------------------------------------------------
    # Unified predict interface
    # ------------------------------------------------------------------

    def _predict(self, clf, X: np.ndarray) -> np.ndarray:
        """
        Returns h(X) in [0,1] regardless of task.
        - Classification: predict_proba[:, 1]
        - Regression: predict, clipped to [0,1]
        """
        if isinstance(clf, ConstantPredictor):
            return clf.predict_proba(X)
        if self.task == "regression":
            return np.clip(clf.predict(X), 0.0, 1.0)
        return clf.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Importance weights (dispatch by constraint)
    # ------------------------------------------------------------------

    def _weights(self, G: np.ndarray, Y: np.ndarray, pi: float) -> np.ndarray:
        if self.constraint == "DP":
            return compute_weights_dp(G, pi, self.p1_)
        else:
            return compute_weights_eo(G, Y, pi, self.p11_, self.p01_)

    def _phi(self, h_pred, G, Y, pi, p1_eval, p11_eval, p01_eval):
        if self.constraint == "DP":
            return phi_dp(h_pred, G, pi, p1_eval)
        return phi_eo(h_pred, G, Y, pi, p11_eval, p01_eval)

    # ------------------------------------------------------------------
    # Weighted ERM for each pi_k
    # ------------------------------------------------------------------

    def _train_component(self, X, G, Y, y, pi_k):
        w = self._weights(G, Y, pi_k)
        w = w / w.mean()  # normalize: mean(w)=1

        if self.model == "mlp":
            w_norm = w / w.sum()
            idx    = np.random.default_rng(0).choice(len(y), size=len(y), replace=True, p=w_norm)
            if self.task == "regression":
                clf = MLPRegressor(
                    hidden_layer_sizes=self.mlp_hidden, max_iter=300,
                    random_state=0, early_stopping=True, validation_fraction=0.1,
                )
            else:
                clf = MLPClassifier(
                    hidden_layer_sizes=self.mlp_hidden, max_iter=300,
                    random_state=0, early_stopping=True, validation_fraction=0.1,
                )
            clf.fit(X[idx], y[idx])

        elif self.task == "regression":
            # Ridge: C is interpreted as 1/alpha
            clf = Ridge(alpha=1.0 / self.C, max_iter=1000)
            clf.fit(X, y, sample_weight=w)

        else:
            clf = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs")
            clf.fit(X, y, sample_weight=w)

        return clf

    # ------------------------------------------------------------------
    # Build constraint matrix A and risk vector
    # ------------------------------------------------------------------

    def _build_A(self, X, G, Y, pi_eval, p1_eval, p11_eval, p01_eval):
        n_clf = len(self.classifiers_)
        A = np.zeros((len(pi_eval), n_clf))
        for k, clf in enumerate(self.classifiers_):
            h_pred = self._predict(clf, X)
            for j, pi_j in enumerate(pi_eval):
                A[j, k] = self._phi(h_pred, G, Y, pi_j, p1_eval, p11_eval, p01_eval)
        return A

    def _risk_vector(self, X, y):
        risks = []
        for clf in self.classifiers_:
            h_pred = self._predict(clf, X)
            if self.task == "regression":
                risks.append(float(np.mean((y - h_pred) ** 2)))
            else:
                proba = np.column_stack([1 - h_pred, h_pred])
                proba = np.clip(proba, 1e-15, 1 - 1e-15)
                risks.append(log_loss(y, proba))
        return np.array(risks)

    def _solve_socp(self, A, r):
        m, K_total = A.shape
        alpha = cp.Variable(K_total)
        prob  = cp.Problem(
            cp.Minimize(r @ alpha),
            [
                alpha >= 0,
                cp.sum(alpha) == 1,
                cp.norm(A @ alpha - self.t * np.ones(m), 2) <= np.sqrt(m * self.delta),
            ]
        )
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"SOCP solver did not converge (status: {prob.status}). "
                "Try increasing delta or adjusting pi_bar."
            )
        return np.array(alpha.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, G: np.ndarray, y: np.ndarray, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p1_ = float(np.mean(G == 1))

        # EO-specific quantities (classification only)
        if self.constraint == "EO":
            pos_mask  = (y == 1)
            self.p11_ = float(np.mean(G[pos_mask] == 1)) if pos_mask.sum() > 0 else 0.5
            self.p01_ = 1.0 - self.p11_
        else:
            self.p11_ = 0.5
            self.p01_ = 0.5

        # Step 1: sample K training perturbation parameters
        self.pi_train_ = self._stratified_sample(self.K, rng)

        # Step 2: train K weighted-ERM predictors
        self.classifiers_ = []
        for pi_k in self.pi_train_:
            self.classifiers_.append(self._train_component(X, G, y, y, pi_k))

        # Step 2b: counterfactual components (classification only)
        if self.add_counterfactual and self.K_counterfactual > 0:
            assert self.task == "classification", \
                "Counterfactual components only supported for classification."
            betas = np.linspace(0.0, 1.0, self.K_counterfactual + 2)[1:-1]
            self.beta_grid_ = betas
            for beta in betas:
                for grp in [1, 0]:
                    y_cf      = y.copy().astype(float)
                    flip_mask = (G == grp)
                    flip      = rng.random(flip_mask.sum()) < beta
                    y_cf[flip_mask] = np.where(flip, 1 - y[flip_mask], y[flip_mask])
                    clf = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs")
                    clf.fit(X, y_cf.astype(int))
                    self.classifiers_.append(clf)

        # Step 2c: constant predictors h≡0 and h≡1
        self.classifiers_.append(ConstantPredictor(0.0))
        self.classifiers_.append(ConstantPredictor(1.0))

        # Step 3: sample m evaluation perturbation parameters
        pi_eval = self._stratified_sample(self.m, rng)

        # Step 4: build constraint matrix A and risk vector
        A = self._build_A(X, G, y, pi_eval, self.p1_, self.p11_, self.p01_)
        r = self._risk_vector(X, y)

        # Step 5: solve SOCP
        self.alpha_ = self._solve_socp(A, r)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns h_alpha(X) in [0,1] (works for both tasks)."""
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        pred = np.zeros(len(X))
        for k, clf in enumerate(self.classifiers_):
            pred += self.alpha_[k] * self._predict(clf, X)
        return np.clip(pred, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions (classification only)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def unfairness_gap(self, X: np.ndarray, G: np.ndarray, Y: np.ndarray = None) -> float:
        pred = self.predict_proba(X)
        if self.constraint == "EO":
            assert Y is not None, "Y required for EO unfairness gap"
            return unfairness_gap_eo(pred, G, Y)
        return unfairness_gap_dp(pred, G)

    def phi_curve(
        self,
        X: np.ndarray,
        G: np.ndarray,
        pi_grid: np.ndarray,
        Y: np.ndarray = None,
    ) -> np.ndarray:
        pred     = self.predict_proba(X)
        p1_eval  = float(np.mean(G == 1))
        if self.constraint == "EO":
            assert Y is not None
            pos      = (Y == 1)
            p11_eval = float(np.mean(G[pos] == 1)) if pos.sum() > 0 else 0.5
            p01_eval = 1.0 - p11_eval
            return np.array([phi_eo(pred, G, Y, pi, p11_eval, p01_eval) for pi in pi_grid])
        return np.array([phi_dp(pred, G, pi, p1_eval) for pi in pi_grid])


# ---------------------------------------------------------------------------
# ERM baseline
# ---------------------------------------------------------------------------

class ERMBaseline:
    """Plain ERM — logistic regression (classification) or Ridge (regression)."""

    def __init__(self, C: float = 1.0, task: str = "classification"):
        self.C    = C
        self.task = task.lower()
        self.clf_ = None
        self.p1_  = None

    def fit(self, X, G, y):
        self.p1_ = float(np.mean(G == 1))
        if self.task == "regression":
            self.clf_ = Ridge(alpha=1.0 / self.C, max_iter=1000)
        else:
            self.clf_ = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs")
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns prediction in [0,1] for both tasks."""
        if self.task == "regression":
            return np.clip(self.clf_.predict(X), 0.0, 1.0)
        return self.clf_.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def unfairness_gap(self, X, G, Y=None, constraint="DP"):
        pred = self.predict_proba(X)
        if constraint == "EO":
            assert Y is not None
            return unfairness_gap_eo(pred, G, Y)
        return unfairness_gap_dp(pred, G)

    def phi_curve(self, X, G, pi_grid, Y=None, constraint="DP"):
        pred    = self.predict_proba(X)
        p1_eval = float(np.mean(G == 1))
        if constraint == "EO":
            assert Y is not None
            pos      = (Y == 1)
            p11_eval = float(np.mean(G[pos] == 1)) if pos.sum() > 0 else 0.5
            p01_eval = 1.0 - p11_eval
            return np.array([phi_eo(pred, G, Y, pi, p11_eval, p01_eval) for pi in pi_grid])
        return np.array([phi_dp(pred, G, pi, p1_eval) for pi in pi_grid])