"""Scoring rules for probability forecasts.

Arrays follow the project convention: `probs` has shape (n, 3) with columns
home/draw/away, and `outcomes` holds 0/1/2 for the result that happened.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Probs = npt.NDArray[np.float64]
Outcomes = npt.NDArray[np.intp]


def log_loss(probs: Probs, outcomes: Outcomes, eps: float = 1e-15) -> float:
    n = len(outcomes)
    if probs.shape != (n, 3):
        raise ValueError(f"probs must have shape ({n},3), got {probs.shape}")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1")
    if not np.isin(outcomes, [0, 1, 2]).all():
        raise ValueError("outcomes must be 0, 1 or 2")
    p_actual = probs[np.arange(n), outcomes]
    p_actual = np.clip(p_actual, eps, 1.0)
    return float(-np.mean(np.log(p_actual)))


def brier_score(probs: Probs, outcomes: Outcomes) -> float:
    n = len(outcomes)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
