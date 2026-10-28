"""Evaluation helpers shared by all experiments.

Every method is evaluated through its randomized prediction function
h(x) = P(yhat = 1 | x) in [0, 1]; accuracy thresholds h at 0.5.
"""

import numpy as np

from .core import (compute_weights_dp, compute_weights_eo, phi_dp, phi_eo,
                   unfairness_gap_dp, unfairness_gap_eo)


def group_stats(G: np.ndarray, y: np.ndarray):
    """(p1, p11, p01) = (P(G=1), P(G=1 | Y=1), P(G=0 | Y=1)) on a split."""
    pos = y == 1
    p1 = float(np.mean(G == 1))
    p11 = float(np.mean(G[pos] == 1)) if pos.any() else 0.5
    return p1, p11, 1.0 - p11


def targets(h_train: np.ndarray, G: np.ndarray, y: np.ndarray):
    """Targets t from ERM's predictions on the training set.

    DP: mean prediction.  EO: average of the two group-conditional TPRs.
    """
    pos = y == 1
    t_dp = float(np.mean(h_train))
    t_eo = 0.5 * (float(np.mean(h_train[pos & (G == 0)]))
                  + float(np.mean(h_train[pos & (G == 1)])))
    return t_dp, t_eo


def phi_ends(h, G, y, constraint, p1, p11, p01):
    """(Phi(h; 0), Phi(h; 1)). Phi is linear in pi', so these two values
    determine the whole curve pi' -> Phi(h; pi')."""
    if constraint == "DP":
        return phi_dp(h, G, 0.0, p1), phi_dp(h, G, 1.0, p1)
    return phi_eo(h, G, y, 0.0, p11, p01), phi_eo(h, G, y, 1.0, p11, p01)


def unfairness(h, G, y, constraint) -> float:
    return unfairness_gap_dp(h, G) if constraint == "DP" else unfairness_gap_eo(h, G, y)


def accuracy(h, y) -> float:
    return float(np.mean((h >= 0.5).astype(int) == y))


def evaluate(h, G, y, t, constraint):
    """Accuracy, U(h), deployment gap and level offset on one test split.

    deploy = max_{pi' in {0, 1}} |Phi(h; pi') - t|   (the sup over [0, 1])
    level  = (Phi(h; 0) + Phi(h; 1)) / 2 - t          (signed)
    """
    p1, p11, p01 = group_stats(G, y)
    phi0, phi1 = phi_ends(h, G, y, constraint, p1, p11, p01)
    return dict(acc=accuracy(h, y), u=unfairness(h, G, y, constraint),
                deploy=max(abs(phi0 - t), abs(phi1 - t)),
                level=0.5 * (phi0 + phi1) - t, phi0=phi0, phi1=phi1, t=t)


def shifted_accuracy(h, G, y, constraint, pi):
    """Importance-weighted accuracy when the deployment proportion is pi."""
    p1, p11, p01 = group_stats(G, y)
    w = (compute_weights_dp(G, pi, p1) if constraint == "DP"
         else compute_weights_eo(G, y, pi, p11, p01))
    correct = ((h >= 0.5).astype(int) == y).astype(float)
    return float(np.sum(w * correct) / np.sum(w))


def worst_case_accuracy(h, G, y, constraint, pi_bar):
    """min over the two ends of [pi_bar, 1 - pi_bar] (accuracy is linear in pi)."""
    return min(shifted_accuracy(h, G, y, constraint, pi_bar),
               shifted_accuracy(h, G, y, constraint, 1.0 - pi_bar))
