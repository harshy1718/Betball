"""Turn bookmaker decimal odds into fair probabilities.

Shape (n, k): one row per match, one column per outcome (3 for 1X2, 2 for over/under).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Odds = npt.NDArray[np.float64]
Probs = npt.NDArray[np.float64]


def _check(odds: Odds) -> None:
    """Raise ValueError unless odds are an (n, k >= 2) table of real prices above 1.0."""
    if odds.ndim != 2 or odds.shape[1] < 2:
        raise ValueError(f"odds must have shape (n, k) with k >= 2, got {odds.shape}")
    if np.isnan(odds).any():
        raise ValueError("odds contain missing prices")
    if (odds <= 1).any():
        raise ValueError("decimal odds must be above 1.0")


def overround(odds: Odds) -> npt.NDArray[np.float64]:
    """The bookmaker's margin per match: sum of 1/odds, minus 1. Written by Harsh."""
    _check(odds)
    probs = 1 / odds
    return probs.sum(axis=1) - 1


def remove_margin(odds: Odds) -> Probs:
    """Fair probabilities (proportional method): 1/odds divided by its row sum. Written by Harsh."""
    _check(odds)
    inv = 1 / odds
    return inv / inv.sum(axis=1, keepdims=True)
