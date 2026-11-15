"""Baseline 2: Elo ratings mapped to H/D/A probabilities.

How it works:
1. Every team has one rating. After each match the winner takes points from the
   loser, more when the result was a surprise (k controls how many).
2. Ratings are updated in date order over the training matches only. For test
   matches the ratings are frozen at the snapshot, so a result inside the test
   week can never move a rating used to predict another test match.
3. A multinomial logistic regression, fitted on training matches, turns the
   pre-match rating difference into P(home), P(draw), P(away).

Teams are rated within their own league (no cross-league matches in the data).
A team with no history starts at the mean of the bottom three teams active in
its league this season or last, minus `promoted_gap`: roughly where promoted
sides belong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from football_edge.models.base import Probs, outcome_index

_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 60.0
    initial_rating: float = 1500.0
    promoted_gap: float = 0.0
    burn_in_seasons: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EloParams:
        return cls(**d)


class Elo:
    name = "elo"

    def __init__(self, params: EloParams | None = None) -> None:
        self.params = params or EloParams()
        self.ratings: dict[str, dict[str, float]] = {}
        self._last_season: dict[str, dict[str, int]] = {}
        self._clf: LogisticRegression | None = None

    # ---- ratings -----------------------------------------------------------
    def _start_rating(
        self, ratings: dict[str, float], last_season: dict[str, int], season: int
    ) -> float:
        active = [r for t, r in ratings.items() if last_season.get(t, -1) >= season - 1]
        if len(active) < 3:
            return self.params.initial_rating
        return float(np.mean(sorted(active)[:3])) - self.params.promoted_gap

    def _ensure(
        self, ratings: dict[str, float], last_season: dict[str, int], team: str, season: int
    ) -> None:
        if team not in ratings:
            ratings[team] = self._start_rating(ratings, last_season, season)

    # ---- Model protocol ----------------------------------------------------
    def fit(self, train: pd.DataFrame) -> None:
        ratings: defaultdict[str, dict[str, float]] = defaultdict(dict)
        last: defaultdict[str, dict[str, int]] = defaultdict(dict)
        order = train.sort_values("date", kind="stable")
        k, hfa = self.params.k, self.params.home_advantage
        diffs = np.empty(len(order))
        rows = order[["league", "season", "home", "away", "FTR"]].itertuples(index=False, name=None)
        for i, (lg, season, home, away, ftr) in enumerate(rows):
            r, ls = ratings[lg], last[lg]
            self._ensure(r, ls, home, season)
            self._ensure(r, ls, away, season)
            d = r[home] + hfa - r[away]
            diffs[i] = d
            expected = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
            change = k * (_SCORE[ftr] - expected)
            r[home] += change
            r[away] -= change
            ls[home] = ls[away] = season
        self.ratings = dict(ratings)
        self._last_season = dict(last)

        first_season = order["season"].min()
        use = (order["season"] >= first_season + self.params.burn_in_seasons).to_numpy()
        y = outcome_index(order["FTR"])[use]
        x = (diffs[use] / 400.0).reshape(-1, 1)
        self._clf = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        if self._clf is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        # Work on copies: predicting must never change the fitted ratings.
        frozen = {lg: dict(t) for lg, t in self.ratings.items()}
        last = {lg: dict(t) for lg, t in self._last_season.items()}
        diffs = []
        rows = test[["league", "season", "home", "away"]].itertuples(index=False, name=None)
        for lg, season, home, away in rows:
            r, ls = frozen.setdefault(lg, {}), last.setdefault(lg, {})
            self._ensure(r, ls, home, season)
            self._ensure(r, ls, away, season)
            diffs.append(r[home] + self.params.home_advantage - r[away])
        x = (np.asarray(diffs, dtype=float) / 400.0).reshape(-1, 1)
        proba = self._clf.predict_proba(x)
        out = np.zeros((len(test), 3))
        out[:, self._clf.classes_] = proba  # classes_ are 0/1/2 = H/D/A
        return out
