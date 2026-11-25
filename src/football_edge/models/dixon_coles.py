"""Dixon-Coles goals model (Dixon & Coles, 1997), adapted for several leagues fitted together
(Premier League, Serie A, La Liga; Bundesliga and Ligue 1 since ADR 0024).

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

Head-to-head (optional, `use_h2h`, ADR 0005): from up to 6 previous meetings of
the two teams in the last 5 years, a goal-difference signal and a total-goals
signal, each shrunk towards zero when there are few meetings. Two coefficients
(gamma_gd, gamma_tot) learn how much they matter; with no real effect they stay
near zero.

Covariates (optional, `covariates`, ADR 0007): named groups of pre-kick-off match
features from data/processed/features.parquet, each with coefficients learned in the fit:
  * "lineup": today's XI value vs the team's usual XI (log ratio). Two coefficients:
    log(lambda_home) += b_att * lineup_home - b_def * lineup_away, mirrored for away.
  * "trust": the same, with the price-free coach's-trust version of XI strength.
  * "manager": 1 in a team's first 6 league games under a new manager. One coefficient:
    log(lambda_home) += k * (new_home - new_away), mirrored for away.
Missing feature values count as 0 (usual XI, no change). Coefficients get the same ridge
penalty as everything else, so with no real effect they stay near zero.

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
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from football_edge.models.base import Probs

if TYPE_CHECKING:
    from football_edge.models.calibration import GoalCalibration

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
    use_h2h: bool = False  # add head-to-head features (ADR 0005)
    h2h_max_meetings: int = 6
    h2h_years: float = 5.0
    h2h_shrink: float = 2.0  # pseudo-meetings of "no signal" added to the denominator
    covariates: tuple[str, ...] = ()  # feature groups from COVARIATE_GROUPS (ADR 0007)
    # Dynamic ratings (ADR 0022): > 0 = a team's attack and defence may change every
    # `dynamic_weeks` weeks, as a random walk whose steps are held back by `rw` (bigger = slower
    # change). 0 = one rating per team (the locked model).
    dynamic_weeks: int = 0
    rw: float = 0.0
    # ADR 0026 B: strength of the pull towards the promoted prior for teams promoted this
    # season, as a share of `ridge` (1.0 = same as everyone, the locked model).
    promoted_pull: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DixonColesParams:
        d = dict(d)
        if "covariates" in d:
            d["covariates"] = tuple(d["covariates"])
            unknown = set(d["covariates"]) - set(COVARIATE_GROUPS)
            if unknown:
                raise ValueError(f"unknown covariate groups: {sorted(unknown)}")
        return cls(**d)


# group -> (home column, away column, coefficient names)
COVARIATE_GROUPS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "lineup": ("lineup_home", "lineup_away", ("lineup_att", "lineup_def")),
    "trust": ("trust_home", "trust_away", ("trust_att", "trust_def")),
    "manager": ("new_mgr_home", "new_mgr_away", ("new_manager",)),
}


def covariate_design(df: pd.DataFrame, groups: tuple[str, ...]) -> tuple[FloatArr, FloatArr]:
    """Design matrices (n, k) adding to log(lambda_home) and log(lambda_away)."""
    hs: list[FloatArr] = []
    as_: list[FloatArr] = []
    for grp in groups:
        h_col, a_col, _ = COVARIATE_GROUPS[grp]
        missing = [c for c in (h_col, a_col) if c not in df.columns]
        if missing:
            raise KeyError(f"covariate group {grp!r} needs columns {missing} (build-features)")
        h = np.nan_to_num(df[h_col].to_numpy(dtype=np.float64))
        a = np.nan_to_num(df[a_col].to_numpy(dtype=np.float64))
        if grp in ("lineup", "trust"):  # own XI on own goals, opponent XI on goals conceded
            hs += [h, -a]
            as_ += [a, -h]
        else:
            hs.append(h - a)
            as_.append(a - h)
    n = len(df)
    xh = np.column_stack(hs) if hs else np.zeros((n, 0))
    xa = np.column_stack(as_) if as_ else np.zeros((n, 0))
    return xh, xa


def tau(x: IntArr, y: IntArr, lam: FloatArr, mu: FloatArr, rho: float) -> FloatArr:
    """Dixon-Coles adjustment factor for low scores (1.0 for every other score)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.asarray(t, dtype=np.float64)


# The grid is built far enough out that the probability left beyond it is below ~1e-9 (at
# least TAIL_GOALS), then every score above max_goals is folded
# into the last row/column, so the edge cell means "max_goals or more". Nothing is thrown away
# and nothing is spread over the other scores (a 12-1 still counts as a home win and as over
# any line the desk uses). With expected goals up to ~4 the mass beyond max_goals is ~0.3%.
TAIL_GOALS = 12


def _reach(lmax: float, max_goals: int) -> int:
    """Goals per side to build so the mass left beyond is negligible (Poisson tail)."""
    return max(TAIL_GOALS, max_goals, int(np.ceil(lmax + 8 * np.sqrt(lmax) + 8)))


class InvalidGridError(ValueError):
    """The low-score correction made a score probability negative: rho is out of range for
    these expected goals. Raised instead of silently clipping."""


def score_grid(lam: float, mu: float, rho: float, max_goals: int = 10) -> FloatArr:
    """P(home goals = i, away goals = j), i, j in 0..max_goals (edge = max_goals or more)."""
    if not (np.isfinite(lam) and np.isfinite(mu)) or lam <= 0 or mu <= 0:
        raise InvalidGridError(f"expected goals must be positive, got {lam}, {mu}")
    factors = (1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho)
    if min(factors) < 0:
        raise InvalidGridError(
            f"rho={rho:.4f} makes a low score negative at expected goals {lam:.2f}-{mu:.2f}"
        )
    g = np.arange(_reach(max(lam, mu), max_goals) + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= factors[0]
    m[0, 1] *= factors[1]
    m[1, 0] *= factors[2]
    m[1, 1] *= factors[3]
    k = max_goals
    out = np.empty((k + 1, k + 1))
    out[:k, :k] = m[:k, :k]
    out[k, :k] = m[k:, :k].sum(axis=0)
    out[:k, k] = m[:k, k:].sum(axis=1)
    out[k, k] = m[k:, k:].sum()
    grid: FloatArr = out / out.sum()
    return grid


def score_grids(lam: FloatArr, mu: FloatArr, rho: float, max_goals: int = 10) -> FloatArr:
    """score_grid for many matches at once: shape (n, max_goals+1, max_goals+1)."""
    lam, mu = np.asarray(lam, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    if len(lam) > 2000:  # keep memory small: the internal grid is ~25 x 25 per match
        parts = [
            score_grids(lam[i : i + 2000], mu[i : i + 2000], rho, max_goals)
            for i in range(0, len(lam), 2000)
        ]
        return np.concatenate(parts)
    if (lam <= 0).any() or (mu <= 0).any() or not np.isfinite(lam + mu).all():
        raise InvalidGridError("expected goals must be positive and finite")
    f = np.stack([1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, np.full_like(lam, 1 - rho)])
    if (f < 0).any():
        raise InvalidGridError(f"rho={rho:.4f} makes a low score negative for some matches")
    g = np.arange(_reach(float(max(lam.max(), mu.max())), max_goals) + 1)
    m = poisson.pmf(g[None, :], lam[:, None])[:, :, None] * poisson.pmf(g[None, :], mu[:, None])[
        :, None, :
    ]
    m[:, 0, 0] *= f[0]
    m[:, 0, 1] *= f[1]
    m[:, 1, 0] *= f[2]
    m[:, 1, 1] *= f[3]
    k = max_goals
    out = np.empty((len(lam), k + 1, k + 1))
    out[:, :k, :k] = m[:, :k, :k]
    out[:, k, :k] = m[:, k:, :k].sum(axis=1)
    out[:, :k, k] = m[:, :k, k:].sum(axis=2)
    out[:, k, k] = m[:, k:, k:].sum(axis=(1, 2))
    grids: FloatArr = out / out.sum(axis=(1, 2), keepdims=True)
    return grids


_EPOCH = pd.Timestamp("2015-07-06")  # a Monday: dynamic rating periods count from here


class DixonColes:
    name = "dixon_coles"

    def __init__(self, params: DixonColesParams | None = None) -> None:
        self.params = params or DixonColesParams()
        self._fitted = False
        # Optional calibration (models/calibration.py, ADR 0020); None = raw model, as in
        # every walk-forward evaluation.
        self.calibration: GoalCalibration | None = None

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

    # ------------------------------------------------------------ head-to-head
    @staticmethod
    def _pair(league: str, a: str, b: str) -> tuple[str, str, str]:
        return (league, a, b) if a < b else (league, b, a)

    def _h2h_from(
        self,
        meetings: list[tuple[pd.Timestamp, str, int, int]],
        home: str,
        date: pd.Timestamp,
        league_avg: float,
    ) -> tuple[float, float]:
        """(gd, tot) signal for `home` from earlier meetings (date, home_team, hg, ag)."""
        cutoff = date - pd.Timedelta(days=365.25 * self.params.h2h_years)
        recent = [m for m in meetings if cutoff <= m[0] < date][-self.params.h2h_max_meetings :]
        if not recent:
            return 0.0, 0.0
        gd = sum((hg - ag) if h == home else (ag - hg) for _, h, hg, ag in recent)
        tot = sum(hg + ag - league_avg for _, _, hg, ag in recent)
        k = len(recent) + self.params.h2h_shrink
        return gd / k, tot / k

    def _h2h_train(self, df: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        """Features for each training match from earlier training matches only (meetings and
        the league's average total so far). Also stores every training meeting, and the full
        training league averages, for use at prediction time."""
        hist: dict[tuple[str, str, str], list[tuple[pd.Timestamp, str, int, int]]] = {}
        goals_so_far: dict[str, list[float]] = {}  # league -> [total goals, matches]
        gd, tot = np.zeros(len(df)), np.zeros(len(df))
        cols = df[["league", "date", "home", "away", "FTHG", "FTAG"]].to_numpy()
        # Process one date at a time: features for a day's matches use only earlier days.
        for _, idx in df.groupby("date", sort=True).indices.items():
            for i in idx:
                lg, date, h, a, _hg, _ag = cols[i]
                past = hist.get(self._pair(lg, h, a), [])
                acc = goals_so_far.get(lg, [0.0, 0.0])
                if self.params.use_h2h and past:
                    avg = acc[0] / acc[1] if acc[1] else 2.6
                    gd[i], tot[i] = self._h2h_from(past, h, date, avg)
            for i in idx:
                lg, date, h, a, hg, ag = cols[i]
                hist.setdefault(self._pair(lg, h, a), []).append((date, h, int(hg), int(ag)))
                acc = goals_so_far.setdefault(lg, [0.0, 0.0])
                acc[0] += float(hg + ag)
                acc[1] += 1.0
        self._h2h_hist = hist
        self._league_avg_total = {lg: v[0] / v[1] for lg, v in goals_so_far.items()}
        return gd, tot

    def _h2h_test(self, test: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        gd, tot = np.zeros(len(test)), np.zeros(len(test))
        if not self.params.use_h2h:
            return gd, tot
        fallback = float(np.mean(list(self._league_avg_total.values())))
        rows = test[["league", "date", "home", "away"]].itertuples(index=False, name=None)
        for i, (lg, date, h, a) in enumerate(rows):
            past = self._h2h_hist.get(self._pair(lg, h, a), [])
            avg = self._league_avg_total.get(lg, fallback)
            gd[i], tot[i] = self._h2h_from(past, h, date, avg)
        return gd, tot

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
        pull = np.ones(n_t)
        latest = df.groupby("league")["season"].max().to_dict()
        for (lg, team), i in self._team_idx.items():
            played = self._seasons_played[(lg, team)]
            cur = int(latest[lg])
            if cur in played and (cur - 1) not in played and cur - 1 >= df["season"].min():
                prior_att[i], prior_def[i] = self._prior_att, self._prior_def
                pull[i] = p.promoted_pull

        lg_i = df["league"].map(self._league_idx).to_numpy()
        h_i = np.array([self._team_idx[k] for k in zip(df.league, df.home, strict=True)])
        a_i = np.array([self._team_idx[k] for k in zip(df.league, df.away, strict=True)])
        team_h, team_a = h_i, a_i
        links = np.zeros((0, 3))  # dynamic: (earlier period, later period, 1 / gap)
        last_of = np.arange(n_t)  # dynamic: each team's latest period
        if p.dynamic_weeks > 0:
            if p.use_h2h or p.covariates:
                raise ValueError("dynamic ratings are not combined with h2h or covariates")
            per = ((df["date"] - _EPOCH).dt.days // (7 * p.dynamic_weeks)).to_numpy()
            pairs = sorted(set(zip(h_i, per, strict=True)) | set(zip(a_i, per, strict=True)))
            tp_idx = {k: i for i, k in enumerate(pairs)}
            h_i = np.array([tp_idx[k] for k in zip(h_i, per, strict=True)])
            a_i = np.array([tp_idx[k] for k in zip(a_i, per, strict=True)])
            # Team-period priors: a spell's first period is pulled towards the team's prior
            # (promoted average if it entered the league after the data starts); every later
            # period is linked to the one before, with a pull weakened by the gap between them
            # (a random walk's variance grows with time).
            per_season = {(int(t), int(q)): int(s) for t, q, s in zip(
                np.r_[team_h, team_a], np.r_[per, per],
                np.r_[df["season"].to_numpy(), df["season"].to_numpy()], strict=True)}  # fmt: skip
            first_season = int(df["season"].min())
            starts, link_rows = [], []
            prev: dict[int, tuple[int, int]] = {}
            for t, q in pairs:
                i = tp_idx[(t, q)]
                s = per_season[(int(t), int(q))]
                if t in prev and s - per_season[(int(t), prev[t][0])] <= 1:
                    link_rows.append((prev[t][1], i, 1.0 / (q - prev[t][0])))
                else:
                    starts.append((i, t, s))
                prev[t] = (int(q), i)
            links = np.array(link_rows, dtype=float).reshape(-1, 3)
            prior_att = np.zeros(len(pairs))
            prior_def = np.zeros(len(pairs))
            start_mask = np.zeros(len(pairs), dtype=bool)
            for i, _t, s in starts:
                start_mask[i] = True
                if s > first_season:
                    prior_att[i], prior_def[i] = self._prior_att, self._prior_def
            last_of = np.array([prev[t][1] for t in range(n_t)])
            n_t = len(pairs)
        x = df["FTHG"].to_numpy(dtype=np.intp)
        y = df["FTAG"].to_numpy(dtype=np.intp)
        tx, ty = self._targets(df, x, y)
        f_gd, f_tot = self._h2h_train(df)
        xh, xa = covariate_design(df, p.covariates)
        nc = self._no_crowd(df["date"])
        age_days = (df["date"].max() - df["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-p.xi * age_days)

        # Parameter vector layout.
        o_mu, o_h, o_c, o_rho = 0, n_l, 2 * n_l, 2 * n_l + 1
        o_att, o_def = 2 * n_l + 2, 2 * n_l + 2 + n_t
        o_g = o_def + n_t  # gamma_gd, gamma_tot (head-to-head)
        o_b = o_g + 2  # covariate coefficients
        n_par = o_b + xh.shape[1]

        is00, is01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        is10, is11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

        def objective(theta: FloatArr) -> tuple[float, FloatArr]:
            mu_l, hfa = theta[o_mu:o_h], theta[o_h:o_c]
            c, rho = theta[o_c], theta[o_rho]
            att, dfn = theta[o_att:o_def], theta[o_def:o_g]
            g_gd, g_tot = theta[o_g], theta[o_g + 1]
            beta = theta[o_b:]
            eta_h = mu_l[lg_i] + hfa[lg_i] + c * nc + att[h_i] - dfn[a_i]
            eta_a = mu_l[lg_i] + att[a_i] - dfn[h_i]
            eta_h = eta_h + g_gd * f_gd + g_tot * f_tot
            eta_a = eta_a - g_gd * f_gd + g_tot * f_tot
            eta_h = eta_h + xh @ beta
            eta_a = eta_a + xa @ beta
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
            grad[o_def:o_g] = -np.bincount(a_i, g_h, minlength=n_t) - np.bincount(
                h_i, g_a, minlength=n_t
            )
            grad[o_g] = np.sum((g_h - g_a) * f_gd)
            grad[o_g + 1] = np.sum((g_h + g_a) * f_tot)
            grad[o_b:] = xh.T @ g_h + xa.T @ g_a

            nll = -float(np.sum(w * ll))
            d_att, d_def = att - prior_att, dfn - prior_def
            if p.dynamic_weeks > 0:  # ridge only where a spell starts; the walk does the rest
                d_att, d_def = d_att * start_mask, d_def * start_mask
            if p.dynamic_weeks == 0 and p.promoted_pull != 1.0:  # ADR 0026 B
                w_att, w_def = pull * d_att, pull * d_def
            else:
                w_att, w_def = d_att, d_def
            pen = p.ridge * (
                np.sum(w_att * d_att) + np.sum(w_def * d_def) + c**2 + g_gd**2 + g_tot**2
                + beta @ beta
            )  # fmt: skip
            g = -grad
            g[o_att:o_def] += 2 * p.ridge * w_att
            g[o_def:o_g] += 2 * p.ridge * w_def
            if len(links):
                i0, i1, wl = links[:, 0].astype(int), links[:, 1].astype(int), links[:, 2]
                for o in (o_att, o_def):
                    d = theta[o + i1] - theta[o + i0]
                    pen += p.rw * float(np.sum(wl * d**2))
                    np.add.at(g, o + i1, 2 * p.rw * wl * d)
                    np.add.at(g, o + i0, -2 * p.rw * wl * d)
            g[o_g] += 2 * p.ridge * g_gd
            g[o_g + 1] += 2 * p.ridge * g_tot
            g[o_c] += 2 * p.ridge * c
            g[o_b:] += 2 * p.ridge * beta
            return nll + pen, g

        theta0 = np.zeros(n_par)
        base = np.log(max((x.mean() + y.mean()) / 2, 0.1))
        theta0[o_mu:o_h] = base
        theta0[o_h:o_c] = 0.25
        theta0[o_att:o_def] = prior_att
        theta0[o_def:o_g] = prior_def
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_par
        bounds[o_rho] = (-0.2, 0.2)
        res = minimize(
            objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
        )
        th = res.x
        self.mu_l, self.hfa = th[o_mu:o_h], th[o_h:o_c]
        self.c, self.rho = float(th[o_c]), float(th[o_rho])
        self.attack, self.defence = th[o_att:o_def][last_of], th[o_def:o_g][last_of]
        if p.dynamic_weeks > 0:  # every period's ratings, for inspection
            self.attack_path, self.defence_path = th[o_att:o_def], th[o_def:o_g]
        self.gamma_gd, self.gamma_tot = float(th[o_g]), float(th[o_g + 1])
        self.beta = th[o_b:]
        names = [n for g in p.covariates for n in COVARIATE_GROUPS[g][2]]
        self.coefficients = dict(zip(names, (float(b) for b in self.beta), strict=True))
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
        f_gd, f_tot = self._h2h_test(test)
        lam, mu = np.empty(len(test)), np.empty(len(test))
        rows = test[["league", "season", "home", "away"]].itertuples(index=False, name=None)
        for k, (lg, season, home, away) in enumerate(rows):
            li = self._league_idx.get(lg)
            base = float(self.mu_l[li]) if li is not None else float(np.mean(self.mu_l))
            hfa = float(self.hfa[li]) if li is not None else float(np.mean(self.hfa))
            ah, dh = self._rating(lg, home, int(season))
            aa, da = self._rating(lg, away, int(season))
            h2h = self.gamma_gd * f_gd[k]
            tot = self.gamma_tot * f_tot[k]
            lam[k] = np.exp(base + hfa + self.c * nc[k] + ah - da + h2h + tot)
            mu[k] = np.exp(base + aa - dh - h2h + tot)
        if self.params.covariates:
            xh, xa = covariate_design(test, self.params.covariates)
            lam, mu = lam * np.exp(xh @ self.beta), mu * np.exp(xa @ self.beta)
        if self.calibration is not None:
            lam, mu = self.calibration.expected(lam, mu)
        return lam, mu

    def raw_expected_goals(self, test: pd.DataFrame) -> tuple[FloatArr, FloatArr]:
        """Expected goals before any calibration (for the log's raw chances)."""
        cal, self.calibration = self.calibration, None
        try:
            return self.expected_goals(test)
        finally:
            self.calibration = cal

    def raw_score_matrix(self, lam: float, mu: float) -> FloatArr:
        """The uncalibrated grid for raw expected goals."""
        return score_grid(lam, mu, self.rho, self.params.max_goals)

    def score_matrix(self, lam: float, mu: float) -> FloatArr:
        """P(home goals = i, away goals = j) for i, j in 0..max_goals (edge = that or more).

        `lam`, `mu` are expected goals as `expected_goals` returns them (calibrated when a
        calibration is attached); the calibration's draw adjustment is applied here."""
        grid = score_grid(lam, mu, self.rho, self.params.max_goals)
        if self.calibration is not None:
            grid = self.calibration.adjust_grid(grid)
        return grid

    def predict_extra(self, test: pd.DataFrame) -> dict[str, FloatArr]:
        """Goals markets from the same score grid: over 2.5, both teams to score."""
        lam, mu = self.expected_goals(test)
        over, btts = np.empty(len(test)), np.empty(len(test))
        g = np.arange(self.params.max_goals + 1)
        total = g[:, None] + g[None, :]
        for k in range(len(test)):
            m = self.score_matrix(float(lam[k]), float(mu[k]))
            over[k] = m[total >= 3].sum()
            btts[k] = m[1:, 1:].sum()
        return {"p_over25": over, "p_btts": btts, "exp_home": lam, "exp_away": mu}

    def predict_proba(self, test: pd.DataFrame) -> Probs:
        lam, mu = self.expected_goals(test)
        out = np.empty((len(test), 3))
        for k in range(len(test)):
            m = self.score_matrix(float(lam[k]), float(mu[k]))
            out[k] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        return out
