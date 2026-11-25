"""One Dixon-Coles model per league, each with its own settings (ADR 0023 challenger).

Each league's model is fitted only on that league's matches, so it gets its own low-score
correction, empty-stadium term and promoted-team prior, plus its own tuned time decay, ridge
and shots weight. Predictions are routed to the model of the match's league.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_edge.models.base import Probs
from football_edge.models.dixon_coles import DixonColes, DixonColesParams

FloatArr = npt.NDArray[np.float64]


class PerLeagueDixonColes:
    name = "dixon_coles_per_league"

    def __init__(self, base: dict[str, Any], by_league: dict[str, dict[str, Any]]) -> None:
        self.base = dict(base)
        self.by_league = {lg: dict(p) for lg, p in by_league.items()}
        self.models: dict[str, DixonColes] = {}

    def params_for(self, league: str) -> DixonColesParams:
        return DixonColesParams.from_dict({**self.base, **self.by_league.get(league, {})})

    def fit(self, train: pd.DataFrame) -> None:
        self.models = {}
        for lg, g in train.groupby("league"):
            m = DixonColes(self.params_for(str(lg)))
            m.fit(g)
            self.models[str(lg)] = m

    @property
    def converged(self) -> bool:
        return all(m.converged for m in self.models.values())

    def _route(self, test: pd.DataFrame) -> dict[str, npt.NDArray[np.intp]]:
        lg = test["league"].astype(str).to_numpy()
        out = {}
        for name in np.unique(lg):
            if name not in self.models:
                raise KeyError(f"no model for league {name!r}: none of its matches in training")
            out[str(name)] = np.flatnonzero(lg == name)
        return out

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        out = np.empty((len(test), 3))
        for lg, idx in self._route(test).items():
            out[idx] = self.models[lg].predict_proba(test.iloc[idx])
        return out

    def predict_extra(self, test: pd.DataFrame) -> dict[str, FloatArr]:
        keys = ("p_over25", "p_btts", "exp_home", "exp_away")
        out = {k: np.empty(len(test)) for k in keys}
        for lg, idx in self._route(test).items():
            part = self.models[lg].predict_extra(test.iloc[idx])
            for k in keys:
                out[k][idx] = part[k]
        return out
