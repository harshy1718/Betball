"""Glicko ratings mapped to H/D/A probabilities (benchmark, ADR 0025).

Like Elo (models/elo.py), but each team also has a rating deviation RD: how unsure the rating
is. RD grows while a team does not play (a summer, a season away) and shrinks with every match,
and an unsure opponent's result moves a rating less. Glickman (1999), Glicko-1, one match per
rating period. A logistic regression on the RD-weighted rating gap gives H/D/A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from football_edge.models.base import Probs, outcome_index

Q = math.log(10) / 400
_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


def g(rd: float) -> float:
    return 1 / math.sqrt(1 + 3 * Q * Q * rd * rd / math.pi**2)


@dataclass(frozen=True)
class GlickoParams:
    home_advantage: float = 60.0
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    min_rd: float = 30.0
    c_per_week: float = 30.0  # RD growth while not playing


class Glicko:
    name = "glicko"

    def __init__(self, params: GlickoParams | None = None) -> None:
        self.params = params or GlickoParams()
        self.state: dict[tuple[str, str], tuple[float, float, pd.Timestamp]] = {}
        self._clf: LogisticRegression | None = None

    def _get(self, key: tuple[str, str], date: pd.Timestamp) -> tuple[float, float]:
        p = self.params
        if key not in self.state:
            return p.initial_rating, p.initial_rd
        r, rd, last = self.state[key]
        weeks = max((date - last).days, 0) / 7
        return r, min(math.sqrt(rd * rd + p.c_per_week**2 * weeks), p.initial_rd)

    def gap(self, lg: str, home: str, away: str, date: pd.Timestamp) -> float:
        rh, dh = self._get((lg, home), date)
        ra, da = self._get((lg, away), date)
        return g(math.sqrt(dh * dh + da * da)) * (rh + self.params.home_advantage - ra) / 400

    def update(self, lg: str, home: str, away: str, date: pd.Timestamp, ftr: str) -> None:
        p = self.params
        rh, dh = self._get((lg, home), date)
        ra, da = self._get((lg, away), date)
        s = _SCORE[ftr]
        sides = (((lg, home), rh, dh, ra, da, s, 1), ((lg, away), ra, da, rh, dh, 1 - s, -1))
        for key, r, rd, ro, rdo, score, sign in sides:
            gg = g(rdo)
            e = 1 / (1 + 10 ** (-gg * (r + sign * p.home_advantage - ro) / 400))
            d2 = 1 / (Q * Q * gg * gg * e * (1 - e))
            denom = 1 / (rd * rd) + 1 / d2
            r_new = r + Q / denom * gg * (score - e)
            self.state[key] = (r_new, max(math.sqrt(1 / denom), p.min_rd), date)

    def fit(self, train: pd.DataFrame) -> None:
        self.state = {}
        order = train.sort_values("date", kind="stable")
        gaps = np.empty(len(order))
        rows = order[["league", "home", "away", "date", "FTR"]].itertuples(index=False, name=None)
        for i, (lg, h, a, d, ftr) in enumerate(rows):
            gaps[i] = self.gap(lg, h, a, pd.Timestamp(d))
            self.update(lg, h, a, pd.Timestamp(d), ftr)
        first = order["season"].min()
        use = (order["season"] > first).to_numpy()  # burn-in: one season
        self._clf = LogisticRegression(C=1e6, max_iter=1000).fit(
            gaps[use].reshape(-1, 1), outcome_index(order["FTR"])[use]
        )

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        if self._clf is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        keys = test[["league", "home", "away", "date"]].itertuples(index=False)
        x = np.array([self.gap(lg, h, a, pd.Timestamp(d)) for lg, h, a, d in keys])
        out = np.zeros((len(test), 3))
        out[:, self._clf.classes_] = self._clf.predict_proba(x.reshape(-1, 1))
        return out
