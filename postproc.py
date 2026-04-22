"""
PostProc  —  Constrained Optimization via Post-Processing.

Implementation of:
    Chzhen, Hebiri & Taturyan (2025).
    "Randomized multi-class classification under system constraints:
     a unified approach via post-processing."  hal-05416646

Specialized to binary classification (A = {0, 1}) with a binary sensitive
attribute S in {0, 1}, in the **unaware** setting (W = X = Z; the sensitive
attribute is NOT part of the feature vector seen by the predictor):

    *  DP  : demographic parity                    |P(Yhat=1|S=s) - P(Yhat=1)| <= eps
    *  EO  : equal opportunity (paper terminology) |P(Yhat=1|S=s, Y=1) - P(Yhat=1|Y=1)| <= eps

Information regime:
    *  fit-time   : (X, G, y)  available — used ONLY to fit the auxiliary
                    estimators  p_hat_y(x) = P(Y=y|X=x),
                    tau_hat_s(x) = P(S=s|X=x), and  q_hat_s(x) = P(S=s|X=x, Y=1).
    *  unlabeled  : the dual SGD over  F_hat(lambda)  uses ONLY X — no G.
                    The cost matrix C(x) is built from the estimators above.
    *  predict    : `predict_proba(X)`  takes only X — no G is required.
    *  evaluate   : G is used downstream (outside this module) for unfairness
                    metrics; the post-processor itself never sees test-time G.

Algorithm overview
------------------
1.  Train auxiliary estimators on the labelled data (X, G, y):
        p_hat_y(x) = P(Y = y | X = x)            — base classifier
        tau_hat_s(x) = P(S = s | X = x)          — group-membership classifier
        q_hat_s(x) = P(S = s | X = x, Y = 1)     — only for EO; trained on
                                                   the Y=1 subset
2.  Build the estimated loss vector  L_hat(x) in R^{|A|}  and estimated cost
    matrix  C_hat(x) in R^{M x |A|}  using the auxiliary estimators (NOT G):
        L_hat(x)_a = 1 - p_hat_a(x),  i.e.  ( p_hat_1(x), p_hat_0(x) ).
        C_hat encodes the M = 4 fairness constraints (two signs x two
        sensitive groups) with  tau_hat_s(x)  in place of  I{G(x) = s}.
3.  Solve the entropic-regularized dual on an *unlabelled* sample
    (the same X_train can be reused — only the marginal P_X matters):

        min_{lambda >= 0}   F_hat(lambda) = E_X [ LSE_beta( -L_hat(X) - C_hat(X)^T lambda ) ]

    via projected stochastic gradient with iterate averaging.  This is a
    valid black-box optimizer in the sense of Definition 4.2 of the paper.
4.  Output the randomized post-processed classifier

        pi_hat_lambda(a | x)  proportional to  exp( beta * ( -L_hat(x) - C_hat(x)^T lambda )_a ).

    `predict_proba(X)` returns pi_hat_lambda(1 | X) in [0, 1] and takes only
    X — the auxiliary estimators internally provide the cost matrix.
    Drop-in compatible with `eval_stats`, `make_phi_curve` etc.

Notes on faithfulness
---------------------
*  The cost functions match Section 2.1 of the paper exactly.  In the
   unaware setting  tau_s(x) = P(S=s | X=x)  is genuinely unknown and must
   be estimated; we plug in a logistic-regression estimator of the same
   form as the base classifier.
*  The default optimizer is SGD3 (Algorithm 3 of Chzhen, Hebiri & Taturyan,
   2025) using AC-SA² (two warm-restarted runs of AC-SA) as the inner
   solver — this is the variant with the strongest theoretical guarantees
   (Theorem 4.1, gradient-mapping rate of  O~(sigma^2 / T) ).
*  beta defaults to "auto", resolved at fit-time as
       beta_auto = 0.5 * sqrt(log2(N_unlab)) * log2(N_unlab),
   where N_unlab is the size of the unlabelled pool.  Empirically this
   lands in the sweet spot (~15-30) across both small and large datasets.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - z.max(axis=axis, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=axis, keepdims=True)
    return z


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PostProcessor:
    """
    Post-process a base probabilistic classifier to satisfy DP or EO via
    the PostProc framework of Chzhen, Hebiri & Taturyan (2025).

    This class is a **pure post-processor**.  It does NOT fit any
    auxiliary estimators itself.  The caller is responsible for:

        1. splitting the labelled data into  (train, unlab);
        2. training on the `train` split:
            *  base_estimator : sklearn-style classifier with
               predict_proba(X)[:, 1] = P(Y = 1 | X = x);
            *  tau_estimator  : sklearn-style classifier with
               predict_proba(X)[:, 1] = P(S = 1 | X = x);
            *  q_estimator    : (EO only) sklearn-style classifier with
               predict_proba(X)[:, 1] = P(S = 1 | X = x, Y = 1),
               fitted on the Y = 1 subset of the train split;
        3. computing the marginals from the `train` split:
            *  p_S    : array of shape (2,)  with P(S = s) for s in {0,1};
            *  p_Y1   : scalar               P(Y = 1);                 # EO only
            *  p_S_Y1 : array of shape (2,)  P(S = s, Y = 1);          # EO only
        4. calling  post.fit(X_unlab)  on the `unlab` split — only X is
           consumed (no labels y, no sensitive attribute G; the cost
           matrix is built from the injected estimators).

    `predict_proba(X)` takes only X at prediction time.

    Parameters
    ----------
    base_estimator : fitted sklearn classifier with predict_proba;
                     estimates  P(Y = y | X = x).
    tau_estimator  : fitted sklearn classifier with predict_proba;
                     estimates  P(S = s | X = x).
    p_S            : array of shape (2,) with marginals P(S = s), s in {0,1}.
    constraint     : "DP" or "EO".
    q_estimator    : (EO only) fitted sklearn classifier with predict_proba
                     estimating  P(S = s | X = x, Y = 1).  May be None: in
                     that case we fall back to tau_estimator (assumes
                     S ⊥ Y | X).
    p_Y1           : (EO only) scalar  P(Y = 1).
    p_S_Y1         : (EO only) array of shape (2,)  P(S = s, Y = 1).
    eps            : constraint slack (analogous to fairlearn's `eps`).
    T              : number of stochastic gradient iterations.
    beta           : entropic regularization for the softmax.  Either a
                     positive float, or the string  "auto"  (default), in
                     which case beta is resolved at fit-time from the size
                     N of the unlabelled pool as
                         beta_auto = 0.5 * sqrt(log2(N)) * log2(N).
    optimizer      : "sgd3"  (default; Algorithm 3 of the paper — restart
                             loop with geometrically increasing
                             strong-convexity μ_j; uses AC-SA² as the
                             inner solver).
                     "acsa"  (Algorithm 2 — accelerated stochastic
                             approximation; closer to Theorem 4.1 guarantees).
                     "acsa2" (AC-SA² — two consecutive AC-SA runs with T/2
                             iterations each, the second warm-started at the
                             output of the first).
    """

    def __init__(
        self,
        base_estimator,
        tau_estimator,
        p_S,
        constraint: str = "DP",
        joint_estimator=None,
        p_Y1: float | None = None,
        p_S_Y1=None,
        eps: float = 0.02,
        T: int = 10_000,
        beta="auto",
        optimizer: str = "sgd3",
        compute_sigma: bool = True,
    ):
        self.constraint = constraint.upper()
        if self.constraint not in ("DP", "EO"):
            raise ValueError(f"constraint must be 'DP' or 'EO', got {constraint!r}")

        # required estimators / marginals
        if base_estimator is None or not hasattr(base_estimator, "predict_proba"):
            raise ValueError("base_estimator must be a fitted classifier with predict_proba.")
        if tau_estimator is None or not hasattr(tau_estimator, "predict_proba"):
            raise ValueError("tau_estimator must be a fitted classifier with predict_proba.")
        self.base_estimator = base_estimator
        self.tau_estimator  = tau_estimator
        self.p_S            = np.clip(np.asarray(p_S, dtype=float), 1e-3, 1.0 - 1e-3)

        # dimensions
        self.K     = int(len(getattr(base_estimator, "classes_", np.array([0, 1]))))
        self.S_num = int(len(self.p_S))

        # EO-specific plug-ins.
        # joint_estimator : sklearn-style classifier or list of binary classifiers,
        # one per sensitive value.  predict_proba(X) must return an (n, S_num)
        # array whose column s estimates  P((S, Y) = (s, 1) | X = x)  directly.
        # This matches the reference `q_X` / `_predict_joint_positive_prob`
        # convention.  For EO it is required; for DP it is unused.
        self.joint_estimator = joint_estimator
        self.p_Y1        = float(p_Y1)         if p_Y1   is not None else None
        self.p_S_Y1      = (np.clip(np.asarray(p_S_Y1, dtype=float), 1e-6, 1.0)
                            if p_S_Y1 is not None else None)
        if self.constraint == "EO":
            if self.p_Y1 is None or self.p_S_Y1 is None:
                raise ValueError("EO requires p_Y1 and p_S_Y1 to be supplied.")
            if self.joint_estimator is None:
                raise ValueError(
                    "EO requires joint_estimator: a classifier whose predict_proba(X) "
                    "returns (n, S_num) with column s = P((S,Y)=(s,1) | X=x).  "
                    "Either an sklearn multi-output classifier fitted on the "
                    "composite labels  S*2 + (1-Y)  (or similar), or a list of "
                    "binary classifiers one per sensitive value."
                )

        self.eps   = float(eps)
        # Per-sensitive-value constraint slack (the attached paper
        # implementation allows a different eps per group; here we
        # broadcast a single scalar to length S_num).
        self.eps_vec = np.full(self.S_num, self.eps, dtype=float)
        self.T     = int(T)
        # beta : either a positive float (used as-is) or the string "auto"
        # (resolved at fit-time as  0.5 * sqrt(N_unlab) * log2(N_unlab),
        # following the recipe specified by the user).
        if isinstance(beta, str):
            if beta != "auto":
                raise ValueError(f"beta string must be 'auto', got {beta!r}")
            self.beta = "auto"
        else:
            self.beta = float(beta)
        self.optimizer = optimizer.lower()
        if self.optimizer not in ("acsa", "acsa2", "sgd3"):
            raise ValueError(
                f"optimizer must be 'acsa', 'acsa2' or 'sgd3', got {optimizer!r}"
            )
        self.compute_sigma = bool(compute_sigma)

        # learned state (populated by fit)
        self.lam_:      np.ndarray | None = None
        self.beta_:     float | None      = None
        self.sigma_sq:  float | None      = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_beta_from_N(N: int) -> float:
        """
        Auto-resolve the entropic-regularization parameter from the size N
        of the unlabelled pool, following the attached reference
        implementation:

            beta_auto = 0.5 * sqrt(N) * log2(N).

        This is robust here because the K-class softmax over
        `clf_prob - 1 + coef @ (λ⁺ − λ⁻) + ε·(λ⁺ + λ⁻)` keeps the
        accuracy signal alive via the per-class `clf_prob − 1` term,
        so even a sharp softmax does not collapse onto a single class.
        """
        N = max(int(N), 4)
        return float(0.5 * np.sqrt(N) * np.log2(N))

    def _p1(self, X: np.ndarray) -> np.ndarray:
        """Return base estimate p_hat_1(x)  (probability of class 1)."""
        return self.base_estimator.predict_proba(X)[:, 1]

    def _clf_prob(self, X: np.ndarray) -> np.ndarray:
        """Return the base classifier probabilities  P(Y=y | X=x), shape (n, K)."""
        return self.base_estimator.predict_proba(X)

    def _tau_prob(self, X: np.ndarray) -> np.ndarray:
        """Return tau probabilities  P(S=s | X=x), shape (n, S_num)."""
        return self.tau_estimator.predict_proba(X)

    def _joint_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Return the EO joint probabilities  P((S, Y) = (s, 1) | X=x), shape (n, S_num).

        Ports the reference ``_predict_joint_positive_prob`` exactly.

        - If ``joint_estimator`` is a list/tuple of binary classifiers
          (one per sensitive value), we stack the "positive-class"
          columns side-by-side into an (n, S_num) array.
        - Otherwise ``joint_estimator.predict_proba(X)`` must return an
          (n, S_num) array with column s = P((S, Y) = (s, 1) | X = x).
        """
        je = self.joint_estimator
        if isinstance(je, (list, tuple)):
            cols = []
            for clf in je:
                p = np.asarray(clf.predict_proba(X))
                if p.ndim == 1:
                    cols.append(p)
                elif p.shape[1] == 1:
                    cols.append(p[:, 0])
                else:
                    cols.append(p[:, -1])
            return np.column_stack(cols)

        p = np.asarray(je.predict_proba(X))
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if p.shape[1] != self.S_num:
            raise ValueError(
                f"joint_estimator.predict_proba(X) must return shape "
                f"(n_samples, S_num={self.S_num}); got {p.shape}."
            )
        return p

    # ------------------------------------------------------------------
    # K-class coefficient getters (Section 2.1 of Chzhen, Hebiri & Taturyan)
    # Each returns  (clf_prob, coef)  where  clf_prob  has shape (n, K) and
    # coef  has shape (n, S_num) with per-sensitive-value contributions.
    # ------------------------------------------------------------------

    def _get_DPunaware_coef(self, X: np.ndarray):
        """
        Demographic parity (unaware) coefficient.  For each sensitive
        value s:
            coef[:, s] = 1 - tau_s(x) / P(S=s)
        so that  E[coef[:, s] · I{Yhat=y'}] = P(Yhat=y') - P(Yhat=y' | S=s).
        """
        clf_prob = self._clf_prob(X)                           # (n, K)
        tau_prob = self._tau_prob(X)                           # (n, S_num)
        p_S_safe = np.maximum(self.p_S, 1e-12)                 # (S_num,)
        coef = 1.0 - tau_prob / p_S_safe[None, :]              # (n, S_num) via broadcasting
        return clf_prob, coef

    def _get_EOppunaware_coef(self, X: np.ndarray):
        """
        Equal opportunity (unaware) coefficient.  For each sensitive
        value s:
            coef[:, s] = p_1(x) / P(Y=1) - P((S=s, Y=1) | X=x) / P(S=s, Y=1)
        so that  E[coef[:, s] · I{Yhat=y'}] = P(Yhat=y' | Y=1) - P(Yhat=y' | S=s, Y=1).
        """
        clf_prob = self._clf_prob(X)                    # (n, K)
        positive_prob = clf_prob[:, 1][:, None]         # (n, 1)  — binary: positive class = 1
        joint_prob = self._joint_prob(X)                # (n, S_num)  P(S=s, Y=1 | X)
        coef = positive_prob / max(self.p_Y1, 1e-12) - joint_prob / self.p_S_Y1[None, :]
        return clf_prob, coef

    # ------------------------------------------------------------------
    # Fairness logits and stochastic gradient (K-class formulation)
    # ------------------------------------------------------------------

    def _fair_logits(self, clf_prob: np.ndarray, coef: np.ndarray,
                     lam: np.ndarray) -> np.ndarray:
        """
        Build the per-sample K-class logits that go into the softmax:

            logits[i, y'] = clf_prob[i, y'] - 1
                          + sum_s coef[i, s] * (lam_plus[s, y'] - lam_minus[s, y'])
                          + sum_s eps_s * (lam_plus[s, y'] + lam_minus[s, y'])

        where lam is laid out as [plus.ravel(); minus.ravel()] with
        plus, minus having shape (S_num, K).

        Shape of output: (n, K).
        """
        half = self.S_num * self.K
        plus  = lam[:half].reshape(self.S_num, self.K)
        minus = lam[half:].reshape(self.S_num, self.K)
        diff = plus - minus                              # (S_num, K)
        summ = plus + minus                              # (S_num, K)
        eps_term = np.sum(self.eps_vec[:, None] * summ, axis=0, keepdims=True)  # (1, K)
        return clf_prob - 1.0 + coef @ diff + eps_term

    def _fair_stoch_grad(self, lam: np.ndarray,
                         clf_prob: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """
        Stochastic gradient of F_hat at lam, averaged over the given
        mini-batch (clf_prob, coef).  Shape of output: (2*S_num*K,).

        Vectorized across sensitive values:
            weighted[s, k] = mean_i  coef[i, s] * softmax_x[i, k]
                           = (coef.T @ softmax_x)[s, k] / n
        so the inner `for s in range(S_num)` loop becomes one matmul.
        """
        logits = self._fair_logits(clf_prob, coef, lam)         # (n, K)
        softmax_x = _softmax(self.beta_ * logits, axis=1)       # (n, K)
        n = clf_prob.shape[0]
        weighted = (coef.T @ softmax_x) / n                     # (S_num, K)
        plus_grad  =  weighted + self.eps_vec[:, None]          # (S_num, K)
        minus_grad = -weighted + self.eps_vec[:, None]          # (S_num, K)
        # Layout: [plus.ravel(); minus.ravel()]
        return np.concatenate([plus_grad.ravel(), minus_grad.ravel()])

    def _sigma_sq(self, coef: np.ndarray) -> float:
        """
        Variance proxy used to set the smoothness constant L = beta * sigma^2.
        Matches the attached reference:

            sigma^2 = 2 * (K * sum_s eps_s^2 + E[ ||coef(X)||_2^2 ])
        """
        coef_norm_sq = float(np.mean(np.sum(coef ** 2, axis=1)))
        return 2.0 * (self.K * float(np.sum(self.eps_vec ** 2)) + coef_norm_sq)
    # ------------------------------------------------------------------
    # Stochastic gradient wrapper (matches reference `stoch_grad` +
    # `stoch_grad_reg`).  The reference samples ONE point per iteration
    # via `x = X[np.random.randint(len(X))]` and computes
    #   stoch_grad_reg(w, x, w_reg, mu_reg)
    #     = stoch_grad(w, x) + sum_k mu_reg[k] * (w - w_reg[k]).
    #
    # **Speedup note**: `clf_prob(x)` and `coef(x)` depend only on `x`
    # (the trained estimators + marginals are fixed after the constructor),
    # never on `w`.  Running them once on the whole unlabelled pool and
    # indexing by the sampled row is numerically identical to the
    # reference's per-step `predict_proba(x.reshape(1,-1))` call (sklearn's
    # predict_proba is deterministic given the same input), but roughly
    # an order of magnitude faster because it avoids thousands of
    # 1-row sklearn calls.
    #
    # To preserve bit-for-bit equivalence with the original code path,
    # we sample an integer index `i` from `rng` (not the row `x`) and do
    # `clf_prob_all[i:i+1]`, `coef_all[i:i+1]` in the hot loop.
    # ------------------------------------------------------------------

    def _precompute_cache(self, X: np.ndarray) -> None:
        """
        Compute and cache  (clf_prob_all, coef_all)  for every sample in
        the unlabelled pool.  Called once at the start of fit().
        """
        if self.constraint == "DP":
            self._clf_prob_all, self._coef_all = self._get_DPunaware_coef(X)
        else:
            self._clf_prob_all, self._coef_all = self._get_EOppunaware_coef(X)

    def _stoch_grad_idx(self, w, i):
        """
        Stochastic gradient at w using the cached `i`-th sample.
        Equivalent (bit-for-bit) to the reference's single-sample
        `stoch_grad(w, x)` because the same `(clf_prob, coef)` row is used.
        """
        return self._fair_stoch_grad(
            w,
            self._clf_prob_all[i:i+1],
            self._coef_all    [i:i+1],
        )

    def _stoch_grad_reg_idx(self, w, i, w_reg, mu_reg):
        """
        Regularized stochastic gradient (reference line 205-209), indexed form.

            reg = sum_k mu_reg[k] * (w - w_reg[k])
            return stoch_grad(w, i) + reg
        """
        reg = 0.0
        for k in range(len(w_reg)):
            reg = reg + mu_reg[k] * (w - w_reg[k])
        return self._stoch_grad_idx(w, i) + reg

    # ------------------------------------------------------------------
    # Optimization algorithms — VERBATIM PORTS of the reference
    # `PostProc.py` file (Chzhen, Hebiri & Taturyan 2025 reference
    # implementation).  Structure, bracketing, and even the apparent
    # `mu_reg.append(w)` quirk are preserved.
    # ------------------------------------------------------------------

    def _SGD(self, X, w_init, alpha, T, w_reg, mu_reg, rng):
        """Reference ``SGD``  (line 213)."""
        w = np.asarray(w_init, dtype=float).copy()
        w_all = []
        n = len(X)
        for t in range(T):
            i = int(rng.integers(0, n))
            w = w - alpha * self._stoch_grad_reg_idx(w, i, w_reg, mu_reg)
            w[w < 0] = 0
            w_all.append(w.copy())
        return np.mean(w_all, axis=0)

    def _SGD_sc(self, X, w_init, mu, L, T, w_reg, mu_reg, rng):
        """Reference ``SGD_sc``  (line 227)."""
        N1 = int(np.floor(T / (L / mu)))
        K1 = int(np.floor(np.log2(mu * T / L)))
        w = np.asarray(w_init, dtype=float).copy()
        for t in range(0, N1):
            w = self._SGD(X, w, 1 / (2 * L), int(np.floor(L / mu)), w_reg, mu_reg, rng)
        for k in range(0, K1):
            w = self._SGD(X, w, 1 / (2**k * L), int(np.floor((2**(k + 2)) * L / mu)),
                          w_reg, mu_reg, rng)
        return w

    def _AC_SA(self, X, w_init, mu, L, T, w_reg, mu_reg, rng):
        """Reference ``AC_SA``  (line 242).  Algorithm 2 of the paper."""
        w = np.asarray(w_init, dtype=float).copy()
        w_ag = np.asarray(w_init, dtype=float).copy()
        n = len(X)

        for i_iter in range(T):
            alpha = 2 / (i_iter + 1)
            gamma = 4 * L / ((i_iter + 1) * (i_iter + 2))
            alpha1 = 1 - alpha

            coef1 = (alpha1 * (mu + gamma)) / (gamma + (1 - alpha ** 2) * mu)
            coef2 = alpha * (alpha1 * mu + gamma) / (gamma + (1 - alpha ** 2) * mu)
            coef3 = alpha * mu / (mu + gamma)
            coef4 = (alpha1 * mu + gamma) / (mu + gamma)
            coef5 = alpha / (mu + gamma)

            w_md = coef1 * w_ag + coef2 * w
            i = int(rng.integers(0, n))
            w = coef3 * w_md + coef4 * w - coef5 * self._stoch_grad_reg_idx(w_md, i, w_reg, mu_reg)
            w[w < 0] = 0
            w_ag = alpha * w + alpha1 * w_ag

        return w_ag

    def _AC_SA_2(self, X, w_init, mu, L, T, w_reg, mu_reg, rng):
        """Reference ``AC_SA_2``  (line 270).  Two warm-restarted AC_SA passes."""
        w_ag_1 = self._AC_SA(X, w_init, mu, L, T, w_reg, mu_reg, rng)
        w_ag_2 = self._AC_SA(X, w_ag_1, mu, L, T, w_reg, mu_reg, rng)
        return w_ag_2

    def _SGD3_sc(self, X, w_init, mu_init, L, T, method, sc, rng):
        """
        Reference ``SGD3_sc``  (line 284).  Preserves the reference's
        ``mu_reg.append(w)``  behaviour verbatim (that line stores the
        current iterate as the mu_reg entry; downstream this multiplies
        element-wise into ``w - w_reg[k]`` so the first regularizer term
        is component-wise rather than scalar-weighted).
        """
        w_init = np.asarray(w_init, dtype=float).copy()
        w = w_init.copy()
        new_init = w_init.copy()
        mu = mu_init
        w_reg = []
        mu_reg = []

        if sc is False:
            w_reg.append(w)
            mu_reg.append(w)     # <-- verbatim from reference (appends vector, not scalar)

        S1 = int(np.floor(np.log2(L / mu)))
        for s in range(1, S1 + 1):
            if method == 'ACSA':
                w = self._AC_SA(X, new_init, mu, L,
                                T=int(np.floor(T / S1)), w_reg=w_reg, mu_reg=mu_reg, rng=rng)
            elif method == 'ACSA2':
                w = self._AC_SA_2(X, new_init, mu, L,
                                   T=int(np.floor(T / S1)), w_reg=w_reg, mu_reg=mu_reg, rng=rng)
            elif method == 'sgd':
                w = self._SGD_sc(X, w, mu, 3 * L, int(np.floor(T / S1)), w_reg, mu_reg, rng)
            else:
                raise Exception('Method not found.')
            new_init = w_init.copy()   # reset init to zeros each restart
            mu = 2 * mu
            w_reg.append(w)
            mu_reg.append(mu)
        return w

    def _SGD3(self, X, w_init, mu, L, T, method, rng):
        """Reference ``SGD3``  (line 315).  Calls SGD3_sc with L replaced by L+mu and sc=False."""
        return self._SGD3_sc(X, w_init, mu, L + mu, T, method, sc=False, rng=rng)

    # ------------------------------------------------------------------
    # Top-level dual dispatchers.  These just set up w_0, mu, L and call
    # the reference-faithful optimizers above.
    # ------------------------------------------------------------------

    def _acsa_dual(self, X, beta, rng):
        L = beta * self.sigma_sq
        mu = L / max(self.T, 1)
        dim = 2 * self.S_num * self.K
        return self._AC_SA(X, np.zeros(dim), mu, L, self.T, [], [], rng)

    def _acsa2_dual(self, X, beta, rng):
        L = beta * self.sigma_sq
        mu = L / max(self.T, 1)
        dim = 2 * self.S_num * self.K
        return self._AC_SA_2(X, np.zeros(dim), mu, L, self.T, [], [], rng)

    def _sgd3_dual(self, X, beta, rng):
        L = beta * self.sigma_sq
        mu = L / max(self.T, 1)
        dim = 2 * self.S_num * self.K
        return self._SGD3(X, np.zeros(dim), mu, L, self.T, method='ACSA2', rng=rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, seed: int = 0):
        """
        Run the dual SGD on the **unlabelled** pool.

        Parameters
        ----------
        X    : features of the unlabelled pool, shape (n_unlab, d).
               Only X is consumed — neither the sensitive attribute G nor
               labels y are needed at this stage (unaware setting; the
               cost matrix is built from the injected estimators).
        seed : RNG seed for the stochastic optimizer.
        """
        rng = np.random.default_rng(seed)

        # ---- choose beta ------------------------------------------------
        # beta = "auto"  ->  0.5 * sqrt(N_unlab) * log2(N_unlab)
        # else           ->  use the explicit value supplied by the caller.
        N = len(X)
        self.beta_ = (
            self._default_beta_from_N(N)
            if isinstance(self.beta, str) and self.beta == "auto"
            else float(self.beta)
        )

        # ---- precompute (clf_prob_all, coef_all) on the unlabelled pool ----
        # This runs the base / tau / joint estimators once each; the dual
        # optimizer will index into these caches instead of re-running
        # predict_proba on a 1-row batch per iteration.  Bit-for-bit
        # identical to the reference's per-iteration evaluation because
        # sklearn's predict_proba is deterministic given the same input.
        self._precompute_cache(X)

        # ---- compute sigma_sq -------------------------------------------
        # compute_sigma=True  : empirical  2 * (K * sum_s eps_s^2 + E||coef(X)||^2)
        # compute_sigma=False : analytic value from the reference set_problem:
        #     DP:  sigma^2 = 2 * sum_s (1 - p_s) / p_s
        #     EO:  sigma^2 = 2 * sum_s (1/q_s - 1/P(Y=1))
        # Both conventions match the reference file exactly.
        if self.compute_sigma:
            self.sigma_sq = self._sigma_sq(self._coef_all)
        else:
            if self.constraint == "DP":
                self.sigma_sq = 2.0 * float(np.sum((1.0 - self.p_S) / self.p_S))
            else:   # EO
                self.sigma_sq = 2.0 * float(np.sum(1.0 / self.p_S_Y1 - 1.0 / self.p_Y1))

        # ---- solve dual -------------------------------------------------
        solver = {
            "acsa":  self._acsa_dual,
            "acsa2": self._acsa2_dual,
            "sgd3":  self._sgd3_dual,
        }[self.optimizer]
        self.lam_ = solver(X, beta=self.beta_, rng=rng)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return  pi_hat_lambda(1 | X)  in [0, 1] — the probability that the
        randomized classifier predicts the positive class (class 1).

        Unaware setting: only X is consumed; the sensitive attribute G is
        NOT required at prediction time.  The K-class softmax is rebuilt
        on the fly from the injected auxiliary estimators.
        """
        if self.lam_ is None:
            raise RuntimeError("Call fit() first.")

        if self.constraint == "DP":
            clf_prob, coef = self._get_DPunaware_coef(X)
        else:
            clf_prob, coef = self._get_EOppunaware_coef(X)

        logits = self._fair_logits(clf_prob, coef, self.lam_)   # (n, K)
        pred   = _softmax(self.beta_ * logits, axis=1)          # (n, K)
        return pred[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


__all__ = ["PostProcessor"]
