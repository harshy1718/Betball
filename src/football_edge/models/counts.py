"""Team count models for corners, yellow cards and red cards.

Same idea as the goals model, for any per-side count (corners won, cards received):

    log E[home count] = mu[league] + home_adv[league] + own[home] + opp[away]
    log E[away count] = mu[league] + own[away] + opp[home]

* own[t]: how many of this the team itself racks up (City win many corners; a rough side
  collects many cards).
* opp[t]: how many its opponents rack up against it (a team that sits deep concedes
  corners; a team that draws fouls gets opponents booked).
Time decay (xi) and ridge shrinkage work exactly as in Dixon-Coles, and a team with no
data gets the league average. With `team_effects=False` it is the league-average baseline
the team model has to beat.

Match totals are rarely exactly Poisson: corners vary more than Poisson allows. The total
is modelled as negative binomial with the team model's mean and a per-league dispersion k
estimated from training totals (method of moments): Var = m + m^2 / k. No extra spread
found -> Poisson.

Only past matches' counts are used; the predicted match's counts are never seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

FloatArr = npt.NDArray[np.float64]

# stat -> (home column, away column, standard lines for "over")
STATS: dict[str, tuple[str, str, tuple[float, ...]]] = {
    "corners": ("HC", "AC", (8.5, 9.5, 10.5, 11.5)),
    "yellows": ("HY", "AY", (3.5, 4.5, 5.5)),
    "reds": ("HR", "AR", (0.5,)),
    # Total cards per side = yellows + reds (each card counts 1). Bookmakers differ on
    # second yellows; football-data records them as a yellow and a red.
    "cards": ("HCARDS", "ACARDS", (3.5, 4.5, 5.5, 6.5)),
}


def with_card_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Add HCARDS/ACARDS (yellows + reds per side) when the card columns are present."""
    if {"HY", "HR", "AY", "AR"} <= set(df.columns) and "HCARDS" not in df.columns:
        df = df.assign(HCARDS=df["HY"] + df["HR"], ACARDS=df["AY"] + df["AR"])
    return df


@dataclass(frozen=True)
class CountParams:
    xi: float = 0.0025  # same time decay as the goals model
    ridge: float = 1.0
    team_effects: bool = True


class TeamCountModel:
    def __init__(self, stat: str, params: CountParams | None = None) -> None:
        if stat not in STATS:
            raise ValueError(f"unknown stat {stat!r} (one of {sorted(STATS)})")
        self.stat = stat
        self.home_col, self.away_col, self.lines = STATS[stat]
        self.params = params or CountParams()
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> None:
        p = self.params
        df = with_card_totals(train)
        df = df.dropna(subset=[self.home_col, self.away_col]).reset_index(drop=True)
        leagues = sorted(df["league"].unique())
        self._league_idx = {lg: i for i, lg in enumerate(leagues)}
        keys = sorted(
            set(zip(df.league, df.home, strict=True)) | set(zip(df.league, df.away, strict=True))
        )
        self._team_idx = {k: i for i, k in enumerate(keys)}
        n_l, n_t = len(leagues), len(keys) if p.team_effects else 0
        lg_i = df["league"].map(self._league_idx).to_numpy()
        h_i = np.array([self._team_idx[k] for k in zip(df.league, df.home, strict=True)])
        a_i = np.array([self._team_idx[k] for k in zip(df.league, df.away, strict=True)])
        x = df[self.home_col].to_numpy(dtype=np.float64)
        y = df[self.away_col].to_numpy(dtype=np.float64)
        age = (df["date"].max() - df["date"]).dt.days.to_numpy(dtype=np.float64)
        w = np.exp(-p.xi * age)
        o_h, o_own, o_opp = n_l, 2 * n_l, 2 * n_l + n_t
        n_par = 2 * n_l + 2 * n_t

        def objective(th: FloatArr) -> tuple[float, FloatArr]:
            mu, hfa = th[:o_h], th[o_h:o_own]
            eta_h = mu[lg_i] + hfa[lg_i]
            eta_a = mu[lg_i].copy()
            if n_t:
                own, opp = th[o_own:o_opp], th[o_opp:]
                eta_h = eta_h + own[h_i] + opp[a_i]
                eta_a = eta_a + own[a_i] + opp[h_i]
            lam, lmu = np.exp(eta_h), np.exp(eta_a)
            nll = -float(np.sum(w * (x * eta_h - lam + y * eta_a - lmu)))
            g_h, g_a = w * (x - lam), w * (y - lmu)
            grad = np.zeros(n_par)
            grad[:o_h] = np.bincount(lg_i, g_h + g_a, minlength=n_l)
            grad[o_h:o_own] = np.bincount(lg_i, g_h, minlength=n_l)
            if n_t:
                grad[o_own:o_opp] = np.bincount(h_i, g_h, minlength=n_t) + np.bincount(
                    a_i, g_a, minlength=n_t
                )
                grad[o_opp:] = np.bincount(a_i, g_h, minlength=n_t) + np.bincount(
                    h_i, g_a, minlength=n_t
                )
            g = -grad
            pen = 0.0
            if n_t:
                eff = th[o_own:]
                pen = p.ridge * float(eff @ eff)
                g[o_own:] += 2 * p.ridge * eff
            return nll + pen, g

        th0 = np.zeros(n_par)
        th0[:o_h] = np.log(max((x.mean() + y.mean()) / 2, 0.01))
        res = minimize(objective, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        th = res.x
        self.mu_l, self.hfa = th[:o_h], th[o_h:o_own]
        self.own = th[o_own:o_opp] if n_t else np.zeros(0)
        self.opp = th[o_opp:] if n_t else np.zeros(0)
        self.converged = bool(res.success)
        self._fitted = True

        # Dispersion of match totals, per league (method of moments, decay-weighted).
        lam, lmu = self._rates(df)
        m, t = lam + lmu, x + y
        self.k: dict[str, float] = {}
        for lg, i in self._league_idx.items():
            sel = lg_i == i
            extra = np.sum(w[sel] * ((t[sel] - m[sel]) ** 2 - m[sel]))
            self.k[lg] = float(np.sum(w[sel] * m[sel] ** 2) / extra) if extra > 0 else float("inf")

    def _rates(self, df: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        lam, mu = np.empty(len(df)), np.empty(len(df))
        rows = df[["league", "home", "away"]].itertuples(index=False, name=None)
        for k, (lg, h, a) in enumerate(rows):
            li = self._league_idx.get(lg)
            base = float(self.mu_l[li]) if li is not None else float(np.mean(self.mu_l))
            hfa = float(self.hfa[li]) if li is not None else float(np.mean(self.hfa))
            hi, ai = self._team_idx.get((lg, h)), self._team_idx.get((lg, a))
            own_h = float(self.own[hi]) if hi is not None and len(self.own) else 0.0
            own_a = float(self.own[ai]) if ai is not None and len(self.own) else 0.0
            opp_h = float(self.opp[hi]) if hi is not None and len(self.opp) else 0.0
            opp_a = float(self.opp[ai]) if ai is not None and len(self.opp) else 0.0
            lam[k] = np.exp(base + hfa + own_h + opp_a)
            mu[k] = np.exp(base + own_a + opp_h)
        return lam, mu

    def expected(self, test: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        """Expected count for the home and the away side."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before predicting")
        return self._rates(test)

    def p_over(self, test: pd.DataFrame, line: float) -> FloatArr:
        """P(match total > line), e.g. line 9.5 -> 10 or more."""
        lam, mu = self.expected(test)
        m = lam + mu
        out = np.empty(len(test))
        for i, lg in enumerate(test["league"]):
            k = self.k.get(lg, float("inf"))
            cut = int(np.floor(line))
            if np.isinf(k):
                out[i] = 1 - poisson.cdf(cut, m[i])
            else:
                out[i] = 1 - nbinom.cdf(cut, k, k / (k + m[i]))
        return out
