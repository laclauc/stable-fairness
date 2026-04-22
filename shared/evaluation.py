"""Shared evaluation helpers — used by all run_*.py scripts."""

import numpy as np
from utils import phi_dp, phi_eo, unfairness_gap_dp, unfairness_gap_eo


def eg_predict_proba(eg_model, X):
    proba = np.zeros(len(X))
    for clf, w in zip(eg_model.predictors_, eg_model.weights_):
        proba += w * clf.predict_proba(X)[:, 1]
    return proba


def test_stats(G_test, y_test):
    """Returns (p1_test, p11_test, p01_test) from a test split."""
    pos = y_test == 1
    p1   = float(np.mean(G_test == 1))
    p11  = float(np.mean(G_test[pos] == 1)) if pos.sum() > 0 else 0.5
    p01  = 1.0 - p11
    return p1, p11, p01


def compute_targets(erm, X_train, G_train, y_train):
    """Compute t_dp and t_eo from an already-fitted ERM on the training set."""
    p = erm.predict_proba(X_train)
    t_dp = float(np.mean(p))
    pos  = y_train == 1
    q0   = float(np.mean(p[pos][G_train[pos] == 0]))
    q1   = float(np.mean(p[pos][G_train[pos] == 1]))
    t_eo = 0.5 * (q0 + q1)
    return t_dp, t_eo


def eval_stats(proba, G_test, y_test, t, p1, p11, p01, constraint):
    """Returns (acc, u, deploy_gap) for a given prediction vector."""
    acc = float(np.mean((proba >= 0.5).astype(int) == y_test))
    if constraint == "DP":
        u      = unfairness_gap_dp(proba, G_test)
        phi0   = phi_dp(proba, G_test, 0.0, p1)
        phi1   = phi_dp(proba, G_test, 1.0, p1)
    else:
        u      = unfairness_gap_eo(proba, G_test, y_test)
        phi0   = phi_eo(proba, G_test, y_test, 0.0, p11, p01)
        phi1   = phi_eo(proba, G_test, y_test, 1.0, p11, p01)
    deploy = max(abs(phi0 - t), abs(phi1 - t))
    return acc, u, deploy


def make_phi_curve(proba, G_test, y_test, pi_grid, p1, p11, p01, constraint):
    """Returns array of shape (len(pi_grid),) with Phi(h; pi') values."""
    if constraint == "DP":
        return np.array([phi_dp(proba, G_test, pi, p1) for pi in pi_grid])
    return np.array([phi_eo(proba, G_test, y_test, pi, p11, p01) for pi in pi_grid])
