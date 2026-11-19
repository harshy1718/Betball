"""Turn bookmaker decimal odds into fair probabilities.

Arrays follow the project convention: shape (n, 3), columns home/draw/away.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Odds = npt.NDArray[np.float64]
Probs = npt.NDArray[np.float64]


def _check(odds: Odds) -> None:
    if odds.ndim != 2 or odds.shape[1] != 3:
        raise ValueError(f"odds must have shape (n, 3), got {odds.shape}")
    if np.isnan(odds).any():
        raise ValueError("odds contain missing prices")
    if (odds <= 1).any():
        raise ValueError("decimal odds must be above 1.0")


def overround(odds: Odds) -> npt.NDArray[np.float64]:
    _check(odds)
    probs = 1 / odds
    return probs.sum(axis=1) - 1


def remove_margin(odds: Odds) -> Probs:
    _check(odds)
    inv = 1 / odds
    return inv / inv.sum(axis=1, keepdims=True)
