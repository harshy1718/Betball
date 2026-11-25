"""Bivariate Poisson score distribution on the Dixon-Coles team ratings (ADR 0025 challenger).

Home goals = X1 + X3 and away goals = X2 + X3, with X1, X2, X3 independent Poisson. The shared
part X3 makes the two scores move together (a lively match lifts both). The means come from the
locked Dixon-Coles fit (same ratings, shots blend, decay, ridge); only the score distribution
changes: instead of the Dixon-Coles low-score correction, one covariance (lambda3) fitted on the
training matches. Karlis & Ntzoufras (2003).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from football_edge.models.dixon_coles import TAIL_GOALS, DixonColes

FloatArr = npt.NDArray[np.float64]
MAX_SHARE = 0.9  # lambda3 at most 90% of the smaller mean, so both private parts stay positive


def _l3(c3: float, lam: FloatArr, mu: FloatArr) -> FloatArr:
    out: FloatArr = np.minimum(c3, MAX_SHARE * np.minimum(lam, mu))
    return out


def bp_pmf(x: npt.NDArray[np.intp], y: npt.NDArray[np.intp], lam: FloatArr, mu: FloatArr,
           l3: FloatArr) -> FloatArr:  # fmt: skip
    """P(home = x, away = y) with means lam, mu and shared part l3, per match."""
    l1, l2 = lam - l3, mu - l3
    out = np.zeros(len(x))
    for k in range(int(np.minimum(x, y).max()) + 1):
        ok = k <= np.minimum(x, y)
        out += np.where(ok, poisson.pmf(x - k, l1) * poisson.pmf(y - k, l2) * poisson.pmf(k, l3), 0)
    return out


def bp_grid(lam: float, mu: float, l3: float, max_goals: int = 10) -> FloatArr:
    """Score grid 0..max_goals (edge = that or more), like dixon_coles.score_grid."""
    n = max(TAIL_GOALS, max_goals, int(np.ceil(max(lam, mu) + 8 * np.sqrt(max(lam, mu)) + 8)))
    g = np.arange(n + 1)
    p1, p2, p3 = poisson.pmf(g, lam - l3), poisson.pmf(g, mu - l3), poisson.pmf(g, l3)
    base = np.outer(p1, p2)
    m = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        if p3[k] < 1e-15:
            break
        m[k:, k:] += p3[k] * base[: n + 1 - k, : n + 1 - k]
    k = max_goals
    out = np.empty((k + 1, k + 1))
    out[:k, :k] = m[:k, :k]
    out[k, :k] = m[k:, :k].sum(axis=0)
    out[:k, k] = m[:k, k:].sum(axis=1)
    out[k, k] = m[k:, k:].sum()
    grid: FloatArr = out / out.sum()
    return grid


class BivariatePoissonDC(DixonColes):
    name = "bivariate_poisson"

    def fit(self, train: pd.DataFrame) -> None:
        super().fit(train)
        df = train.reset_index(drop=True)
        lam, mu = self.expected_goals(df)
        x, y = df["FTHG"].to_numpy(int), df["FTAG"].to_numpy(int)
        age = (df["date"].max() - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.params.xi * age)

        def nll(c3: float) -> float:
            p = bp_pmf(x, y, lam, mu, _l3(c3, lam, mu))
            return -float(np.sum(w * np.log(np.clip(p, 1e-300, None))))

        self.c3 = float(minimize_scalar(nll, bounds=(0.0, 0.4), method="bounded").x)

    def score_matrix(self, lam: float, mu: float) -> FloatArr:
        l3 = float(_l3(self.c3, np.array([lam]), np.array([mu]))[0])
        return bp_grid(lam, mu, l3, self.params.max_goals)
