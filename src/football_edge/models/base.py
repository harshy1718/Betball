"""The contract every model follows, so the evaluator can treat them all the same."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

# Column order of every probability array in the project: home win, draw, away win.
OUTCOMES = ("H", "D", "A")

Probs = npt.NDArray[np.float64]  # shape (n_matches, 3), rows sum to 1


class Model(Protocol):
    """A 1X2 model. `fit` sees only training matches; `predict_proba` must not
    read result columns (FTHG, FTAG, FTR) of the matches it predicts."""

    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict_proba(self, test: pd.DataFrame) -> Probs: ...


def outcome_index(ftr: pd.Series) -> npt.NDArray[np.intp]:
    """Map full-time results H/D/A to 0/1/2 (the column order of `Probs`)."""
    mapping = {o: i for i, o in enumerate(OUTCOMES)}
    idx = ftr.map(mapping)
    if idx.isna().any():
        raise ValueError(f"unexpected result values: {sorted(set(ftr) - set(OUTCOMES))}")
    return np.asarray(idx, dtype=np.intp)
