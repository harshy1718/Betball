"""Turn bookmaker decimal odds into fair probabilities.

Arrays follow the project convention: shape (n, 3), columns home/draw/away.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Odds = npt.NDArray[np.float64]
Probs = npt.NDArray[np.float64]


def overround(odds: Odds) -> npt.NDArray[np.float64]:
    """>>> HARSH WRITES THIS <<<  Spec: docs/A4_1_margin_spec.md.

    The bookmaker's margin per match: sum of 1/odds, minus 1.
    """
    raise NotImplementedError("A4.1: write overround (see docs/A4_1_margin_spec.md)")


def remove_margin(odds: Odds) -> Probs:
    """>>> HARSH WRITES THIS <<<  Spec: docs/A4_1_margin_spec.md.

    Fair probabilities: each 1/odds divided by the row's sum of 1/odds.
    """
    raise NotImplementedError("A4.1: write remove_margin (see docs/A4_1_margin_spec.md)")
