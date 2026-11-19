"""Dixon-Coles goals model (Dixon & Coles, 1997), adapted for three leagues.

The idea: predict how many goals each side scores, then turn the goal
distribution into home/draw/away probabilities.

Expected goals for one match (log scale):

    log(lambda_home) = mu[league] + home_adv[league] + no_crowd * c
                       + attack[home] - defence[away]
    log(lambda_away) = mu[league] + attack[away] - defence[home]

* mu: the league's baseline scoring rate.
* home_adv: the league's home advantage; c adjusts it for the empty-stadium
  period (COVID), when home advantage largely disappeared.
* attack / defence: one pair per team, rated within its own league.

Goals are Poisson, with the Dixon-Coles correction `rho` for the four
low scores (0-0, 1-0, 0-1, 1-1), which plain Poisson gets slightly wrong.

Shots (optional, `shots_weight`): goals are rare and noisy, so team ratings can
be learned from a blend of actual goals and "shot-based goals" (shots on target
x the league's goals-per-shot-on-target rate). 0 = goals only, 1 = shots only.
Only past matches' shots are used; the predicted match's stats are never seen.

Three things keep it honest and stable:
1. Time decay: a match `t` days before the latest training date gets weight
   exp(-xi * t), so recent form counts more than results from years ago.
2. Shrinkage (ridge): ratings are pulled towards a prior, so a team with few
   matches is not given an extreme rating from a handful of results.
3. Promoted teams: a team new to its league is pulled towards the average
   promoted side (estimated from earlier seasons in the training data), not
   towards the league average. A team with no data at all gets that prior.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from football_edge.models.base import Probs

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.intp]


@dataclass(frozen=True)
class DixonColesParams:
    xi: float = 0.0019  # time decay per day (half-life about one year)
    ridge: float = 2.0  # strength of the pull towards the prior
    max_goals: int = 10  # score grid is 0..max_goals for each side
    no_crowd_start: str = "2020-06-01"
    no_crowd_end: str = "2021-05-31"
    min_promoted_matches: int = 5  # matches needed for a team-season to inform the prior
    shots_weight: float = 0.0  # 0 = learn from goals only, 1 = from shots on target only

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DixonColesParams:
        return cls(**d)


def tau(x: IntArr, y: IntArr, lam: FloatArr, mu: FloatArr, rho: float) -> FloatArr:
    """Dixon-Coles adjustment factor for low scores (1.0 for every other score)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.asarray(t, dtype=np.float64)


class DixonColes:
    name = "dixon_coles"

    def __init__(self, params: DixonColesParams | None = None) -> None:
        self.params = params or DixonColesParams()
        self._fitted = False

    # ------------------------------------------------------------------ helpers
    def _no_crowd(self, dates: pd.Series) -> FloatArr:
        start = pd.Timestamp(self.params.no_crowd_start)
        end = pd.Timestamp(self.params.no_crowd_end)
        return np.asarray(dates.between(start, end), dtype=np.float64)

    def _promoted_prior(self, train: pd.DataFrame) -> tuple[float, float]:
        """Average (attack, defence) of promoted sides in their first season, as log ratios
        against their league's average, from completed history in the training data."""
        att, dfn = [], []
        for _lg, g in train.groupby("league"):
            seasons = sorted(g["season"].unique())
            teams = {
                s: set(g.loc[g.season == s, "home"]) | set(g.loc[g.season == s, "away"])
                for s in seasons
            }
            for prev, cur in itertools.pairwise(seasons):
                if cur != prev + 1:
                    continue
                gs = g[g.season == cur]
                league_rate = (gs["FTHG"].sum() + gs["FTAG"].sum()) / (2 * len(gs))
                for team in teams[cur] - teams[prev]:
                    home = gs[gs.home == team]
                    away = gs[gs.away == team]
                    n = len(home) + len(away)
                    if n < self.params.min_promoted_matches:
                        continue
                    scored = (home["FTHG"].sum() + away["FTAG"].sum()) / n
                    conceded = (home["FTAG"].sum() + away["FTHG"].sum()) / n
                    att.append(np.log(max(scored, 0.05) / league_rate))
                    dfn.append(-np.log(max(conceded, 0.05) / league_rate))
        if not att:
            return 0.0, 0.0
        return float(np.mean(att)), float(np.mean(dfn))

    def _targets(self, df: pd.DataFrame, x: IntArr, y: IntArr) -> tuple[FloatArr, FloatArr]:
        """Goals, or a blend of goals and shot-based goals, per side of each match."""
        sw = self.params.shots_weight
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        if sw == 0.0 or not {"HST", "AST"} <= set(df.columns):
            return xf, yf
        hst = df["HST"].to_numpy(dtype=np.float64)
        ast = df["AST"].to_numpy(dtype=np.float64)
        ok = ~(np.isnan(hst) | np.isnan(ast))
        conv = {}
        for lg, g in df[ok].groupby("league"):
            sot = g["HST"].sum() + g["AST"].sum()
            conv[lg] = (g["FTHG"].sum() + g["FTAG"].sum()) / sot if sot > 0 else 0.0
        rate = df["league"].map(conv).fillna(0.0).to_numpy(dtype=np.float64)
        tx = np.where(ok, (1 - sw) * xf + sw * np.nan_to_num(hst) * rate, xf)
        ty = np.where(ok, (1 - sw) * yf + sw * np.nan_to_num(ast) * rate, yf)
        return tx, ty

    # ---------------------------------------------------------------- fitting
    def fit(self, train: pd.DataFrame) -> None:
        p = self.params
        df = train.reset_index(drop=True)

        leagues = sorted(df["league"].unique())
        self._league_idx = {lg: i for i, lg in enumerate(leagues)}
        keys = sorted(
            set(zip(df.league, df.home, strict=True)) | set(zip(df.league, df.away, strict=True))
        )
        self._team_idx = {k: i for i, k in enumerate(keys)}
        n_l, n_t = len(leagues), len(keys)

        # Which teams count as "active" in which season (for the newcomer prior).
        self._seasons_played: dict[tuple[str, str], set[int]] = {}
        for lg, s, h, a in df[["league", "season", "home", "away"]].itertuples(index=False):
            self._seasons_played.setdefault((lg, h), set()).add(int(s))
            self._seasons_played.setdefault((lg, a), set()).add(int(s))

        self._prior_att, self._prior_def = self._promoted_prior(df)
        prior_att = np.zeros(n_t)
        prior_def = np.zeros(n_t)
        latest = df.groupby("league")["season"].max().to_dict()
        for (lg, team), i in self._team_idx.items():
            played = self._seasons_played[(lg, team)]
            cur = int(latest[lg])
            if cur in played and (cur - 1) not in played and cur - 1 >= df["season"].min():
                prior_att[i], prior_def[i] = self._prior_att, self._prior_def

        lg_i = df["league"].map(self._league_idx).to_numpy()
        h_i = np.array([self._team_idx[k] for k in zip(df.league, df.home, strict=True)])
        a_i = np.array([self._team_idx[k] for k in zip(df.league, df.away, strict=True)])
        x = df["FTHG"].to_numpy(dtype=np.intp)
        y = df["FTAG"].to_numpy(dtype=np.intp)
        tx, ty = self._targets(df, x, y)
        nc = self._no_crowd(df["date"])
        age_days = (df["date"].max() - df["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-p.xi * age_days)

        # Parameter vector layout.
        o_mu, o_h, o_c, o_rho = 0, n_l, 2 * n_l, 2 * n_l + 1
        o_att, o_def = 2 * n_l + 2, 2 * n_l + 2 + n_t
        n_par = o_def + n_t

        is00, is01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        is10, is11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

        def objective(theta: FloatArr) -> tuple[float, FloatArr]:
            mu_l, hfa = theta[o_mu:o_h], theta[o_h:o_c]
            c, rho = theta[o_c], theta[o_rho]
            att, dfn = theta[o_att:o_def], theta[o_def:]
            eta_h = mu_l[lg_i] + hfa[lg_i] + c * nc + att[h_i] - dfn[a_i]
            eta_a = mu_l[lg_i] + att[a_i] - dfn[h_i]
            lam, mu = np.exp(eta_h), np.exp(eta_a)
            t = np.maximum(tau(x, y, lam, mu, rho), 1e-10)

            # Poisson terms use the (possibly shot-blended) targets; the low-score
            # correction always uses the actual score.
            ll = np.log(t) + tx * eta_h - lam + ty * eta_a - mu
            # d log-lik / d eta_h, d eta_a, d rho for each match
            g_h = tx - lam
            g_a = ty - mu
            g_rho = np.zeros_like(lam)
            g_h = g_h + np.where(is00, -lam * mu * rho / t, 0) + np.where(is01, lam * rho / t, 0)
            g_a = g_a + np.where(is00, -lam * mu * rho / t, 0) + np.where(is10, mu * rho / t, 0)
            g_rho = (
                np.where(is00, -lam * mu / t, 0)
                + np.where(is01, lam / t, 0)
                + np.where(is10, mu / t, 0)
                + np.where(is11, -1.0 / t, 0)
            )
            g_h, g_a, g_rho = w * g_h, w * g_a, w * g_rho

            grad = np.zeros(n_par)
            grad[o_mu:o_h] = np.bincount(lg_i, g_h + g_a, minlength=n_l)
            grad[o_h:o_c] = np.bincount(lg_i, g_h, minlength=n_l)
            grad[o_c] = np.sum(g_h * nc)
            grad[o_rho] = np.sum(g_rho)
            grad[o_att:o_def] = np.bincount(h_i, g_h, minlength=n_t) + np.bincount(
                a_i, g_a, minlength=n_t
            )
            grad[o_def:] = -np.bincount(a_i, g_h, minlength=n_t) - np.bincount(
                h_i, g_a, minlength=n_t
            )

            nll = -float(np.sum(w * ll))
            d_att, d_def = att - prior_att, dfn - prior_def
            pen = p.ridge * (np.sum(d_att**2) + np.sum(d_def**2) + c**2)
            g = -grad
            g[o_att:o_def] += 2 * p.ridge * d_att
            g[o_def:] += 2 * p.ridge * d_def
            g[o_c] += 2 * p.ridge * c
            return nll + pen, g

        theta0 = np.zeros(n_par)
        base = np.log(max((x.mean() + y.mean()) / 2, 0.1))
        theta0[o_mu:o_h] = base
        theta0[o_h:o_c] = 0.25
        theta0[o_att:o_def] = prior_att
        theta0[o_def:] = prior_def
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_par
        bounds[o_rho] = (-0.2, 0.2)
        res = minimize(
            objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
        )
        th = res.x
        self.mu_l, self.hfa = th[o_mu:o_h], th[o_h:o_c]
        self.c, self.rho = float(th[o_c]), float(th[o_rho])
        self.attack, self.defence = th[o_att:o_def], th[o_def:]
        self.converged = bool(res.success)
        self._fitted = True

    # -------------------------------------------------------------- predicting
    def _rating(self, league: str, team: str, season: int) -> tuple[float, float]:
        played = self._seasons_played.get((league, team), set())
        if season in played or (season - 1) in played:
            i = self._team_idx[(league, team)]
            return float(self.attack[i]), float(self.defence[i])
        return self._prior_att, self._prior_def  # newly promoted or unseen

    def expected_goals(self, test: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predicting")
        nc = self._no_crowd(test["date"])
        lam, mu = np.empty(len(test)), np.empty(len(test))
        rows = test[["league", "season", "home", "away"]].itertuples(index=False, name=None)
        for k, (lg, season, home, away) in enumerate(rows):
            li = self._league_idx.get(lg)
            base = float(self.mu_l[li]) if li is not None else float(np.mean(self.mu_l))
            hfa = float(self.hfa[li]) if li is not None else float(np.mean(self.hfa))
            ah, dh = self._rating(lg, home, int(season))
            aa, da = self._rating(lg, away, int(season))
            lam[k] = np.exp(base + hfa + self.c * nc[k] + ah - da)
            mu[k] = np.exp(base + aa - dh)
        return lam, mu

    def score_matrix(self, lam: float, mu: float) -> FloatArr:
        """P(home goals = i, away goals = j) for i, j in 0..max_goals, normalised."""
        g = np.arange(self.params.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        out: FloatArr = m / m.sum()
        return out

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        lam, mu = self.expected_goals(test)
        out = np.empty((len(test), 3))
        for k in range(len(test)):
            m = self.score_matrix(float(lam[k]), float(mu[k]))
            out[k] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        return out
