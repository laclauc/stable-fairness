"""
Helper that prepares a `PostProcessor` for a fairness experiment:

    1. splits the labelled data into `train` and `unlab`;
    2. fits  base_estimator (P(Y=1|X)),  tau_estimator (P(S=1|X)),  and
       (for EO)  q_estimator (P(S=1|X, Y=1))  on the train split;
    3. computes the marginals  P(S=s),  P(Y=1),  P(S=s, Y=1)  on train;
    4. returns  (post, X_ul)  where  `post`  is an unfitted
       `PostProcessor`  ready for  ``post.fit(X_ul, seed=...)``.

This isolates the orchestration in a single place so each `run_*.py`
script can reproduce the unaware-setting protocol with a one-liner:

    post, X_ul = make_postproc(
        X_tr, G_tr, y_tr, constraint="DP", eps=0.01, T=10_000, seed=seed
    )
    post.fit(X_ul, seed=seed)
    proba_te = post.predict_proba(X_te)

G of the unlabelled split is intentionally NOT returned: `PostProcessor.fit`
only consumes X (the cost matrix is built from the injected estimators),
matching the unaware-setting information regime exactly.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

from postproc import PostProcessor


def split_train_unlab(
    X: np.ndarray, G: np.ndarray, y: np.ndarray,
    train_frac: float = 0.5, seed: int = 0,
):
    """Random 50/50 split of (X, G, y) into a labelled `train` and an
    unlabelled `unlab` (the unlab portion's labels and sensitive attribute
    are not used by the post-processor itself; only X_ul is consumed).
    """
    n    = len(y)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k    = int(round(train_frac * n))
    tr, ul = perm[:k], perm[k:]
    return (X[tr], G[tr], y[tr]), (X[ul], G[ul], y[ul])


def fit_estimators(
    X_tr: np.ndarray, G_tr: np.ndarray, y_tr: np.ndarray,
    constraint: str = "DP", base_C: float = 1.0,
):
    """
    Fit the three auxiliary estimators on the train split and compute the
    relevant marginals.

    Returns a dict with keys
        base_estimator, tau_estimator, q_estimator (None unless EO),
        p_S, p_Y1 (None unless EO), p_S_Y1 (None unless EO).
    """
    G_tr = np.asarray(G_tr).astype(int)
    y_tr = np.asarray(y_tr).astype(int)
    if len(np.unique(G_tr)) < 2:
        raise ValueError("Sensitive attribute G must take both values 0 and 1.")

    # base : P(Y = 1 | X)
    base = LogisticRegression(C=base_C, max_iter=1000, solver="lbfgs")
    base.fit(X_tr, y_tr)

    # tau : P(S = 1 | X)
    tau = LogisticRegression(C=base_C, max_iter=1000, solver="lbfgs")
    tau.fit(X_tr, G_tr)

    # q (EO only) : P(S = 1 | X, Y = 1)
    q = None
    if constraint.upper() == "EO":
        mask = (y_tr == 1)
        if mask.sum() >= 10 and len(np.unique(G_tr[mask])) == 2:
            q = LogisticRegression(C=base_C, max_iter=1000, solver="lbfgs")
            q.fit(X_tr[mask], G_tr[mask])
        # else: leave None — PostProcessor falls back to tau.

    # marginals
    p_S = np.array([float(np.mean(G_tr == 0)), float(np.mean(G_tr == 1))])
    if constraint.upper() == "EO":
        p_Y1   = float(np.mean(y_tr == 1))
        p_S_Y1 = np.array([float(np.mean((G_tr == 0) & (y_tr == 1))),
                           float(np.mean((G_tr == 1) & (y_tr == 1)))])
    else:
        p_Y1   = None
        p_S_Y1 = None

    return dict(base_estimator=base, tau_estimator=tau, q_estimator=q,
                p_S=p_S, p_Y1=p_Y1, p_S_Y1=p_S_Y1)


def make_postproc(
    X: np.ndarray, G: np.ndarray, y: np.ndarray,
    constraint: str = "DP", eps: float = 0.02, T: int = 10_000,
    optimizer: str = "sgd", beta=None, eta0: float = 1.0,
    base_C: float = 1.0, train_frac: float = 0.5, seed: int = 0,
):
    """
    One-call orchestration helper.  Splits, trains the estimators, then
    returns  (post, X_ul)  where  ``post``  is an **unfitted**
    `PostProcessor`.  Caller still does
        post.fit(X_ul, seed=seed)
    so the dual-SGD step is explicit and the seed is reusable.
    """
    (X_tr, G_tr, y_tr), (X_ul, _G_ul, _y_ul) = split_train_unlab(
        X, G, y, train_frac=train_frac, seed=seed,
    )
    est = fit_estimators(X_tr, G_tr, y_tr, constraint=constraint, base_C=base_C)
    post = PostProcessor(
        constraint=constraint, eps=eps, T=T,
        optimizer=optimizer, beta=beta, eta0=eta0,
        **est,
    )
    return post, X_ul


__all__ = ["split_train_unlab", "fit_estimators", "make_postproc"]
