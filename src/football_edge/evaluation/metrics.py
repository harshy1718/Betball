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
    """>>> HARSH WRITES THIS <<<  Spec: docs/A1_4_log_loss_spec.md.

    Mean negative log probability given to the outcome that happened.
    """
    raise NotImplementedError("A1.4: write log_loss (see docs/A1_4_log_loss_spec.md)")


def brier_score(probs: Probs, outcomes: Outcomes) -> float:
    """Mean squared distance between the forecast and the one-hot result.

    0 is perfect; always predicting (1/3, 1/3, 1/3) scores 2/3.
    """
    n = len(outcomes)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
