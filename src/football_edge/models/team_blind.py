"""Baseline 1: ignores who is playing. Predicts each league's historical H/D/A rates.

Any real model must beat this, or it has learned nothing about teams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.models.base import OUTCOMES, Probs


class TeamBlind:
    name = "team_blind"

    def __init__(self) -> None:
        self._rates: dict[str, np.ndarray] = {}
        self._overall: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> None:
        def rates(results: pd.Series) -> np.ndarray:
            counts = results.value_counts().reindex(list(OUTCOMES), fill_value=0)
            return counts.to_numpy(dtype=float) / counts.sum()

        self._overall = rates(train["FTR"])
        self._rates = {str(lg): rates(g["FTR"]) for lg, g in train.groupby("league")}

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        if self._overall is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        overall = self._overall
        rows = [self._rates.get(str(lg), overall) for lg in test["league"]]
        return np.vstack(rows) if rows else np.empty((0, 3))
