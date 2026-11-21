"""Scoring rules for probability forecasts.

`probs` has shape (n, k): one row per match, one column per possible outcome
(k = 3 for home/draw/away, k = 2 for over/under). `outcomes` holds the column
number of what happened.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Probs = npt.NDArray[np.float64]
Outcomes = npt.NDArray[np.intp]


def log_loss(probs: Probs, outcomes: Outcomes, eps: float = 1e-15) -> float:
    """Mean of -ln(probability given to the actual result); lower is better.

    Used instead of accuracy because a bet depends on probability x odds, so the
    confidence itself must be accurate; log-loss punishes confident misses hardest.
    Written by Harsh (spec: docs/A1_4_log_loss_spec.md).
    """
    n = len(outcomes)
    if probs.ndim != 2 or probs.shape[0] != n or probs.shape[1] < 2:
        raise ValueError(f"probs must have shape ({n}, k) with k >= 2, got {probs.shape}")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1")
    if not np.isin(outcomes, np.arange(probs.shape[1])).all():
        raise ValueError(f"outcomes must be column numbers 0..{probs.shape[1] - 1}")
    p_actual = probs[np.arange(n), outcomes]
    p_actual = np.clip(p_actual, eps, 1.0)
    return float(-np.mean(np.log(p_actual)))


def brier_score(probs: Probs, outcomes: Outcomes) -> float:
    n = len(outcomes)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
