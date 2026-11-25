"""Calibration of the goals and cards probabilities (ADR 0020).

The raw models are right on average but bent at the edges: totals spread too wide, clear
favourites win more often than predicted, draws a little too rare, cards over-spread. Fixing each
market separately would make single legs disagree with the same-match parlays built from the
same score grid, so the calibration works on the model's own quantities instead:

Goals (five numbers). With s = log lambda - log mu (who is stronger) and t = log lambda +
log mu (how many goals):
    s' = a*s + b                     a > 1 stretches the gap between the sides
    t' = t_ref + d*(t - t_ref) + c   d < 1 pulls every match's total towards the average
and the score grid's draw cells are multiplied by exp(delta), then the grid is renormalised.
Every goals market, leg and parlay is then read from one calibrated grid.

Cards (two numbers). The expected match total is pulled towards the league-average model:
m' = base^(1-d) * m^d, and the negative-binomial dispersion is scaled, k' = f*k.

Both are fitted only on out-of-sample predictions of earlier seasons (walk-forward), by
minimising log loss: goals on match result + over/under 2.5 + both-score, cards on the four
standard lines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

FloatArr = npt.NDArray[np.float64]
EPS = 1e-12
CARD_LINES = (3.5, 4.5, 5.5, 6.5)


def adjust_draws(grids: FloatArr, delta: float | FloatArr) -> FloatArr:
    """Multiply the draw cells of one grid (2-D) or many (3-D) by exp(delta); renormalise."""
    g = np.array(grids, dtype=np.float64, copy=True)
    n = g.shape[-1]
    idx = np.arange(n)
    if g.ndim == 2:
        g[idx, idx] *= np.exp(float(np.asarray(delta)))
        out: FloatArr = g / g.sum()
        return out
    d = np.broadcast_to(np.asarray(delta, dtype=np.float64), (g.shape[0],))
    g[:, idx, idx] *= np.exp(d)[:, None]
    out = g / g.sum(axis=(1, 2), keepdims=True)
    return out


@dataclass(frozen=True)
class GoalCalibration:
    a: float = 1.0
    b: float = 0.0
    d: float = 1.0
    c: float = 0.0
    delta: float = 0.0
    t_ref: float = 0.0
    fitted_on: str = ""  # e.g. "2018-2025, 9,120 matches"

    def expected(self, lam: FloatArr, mu: FloatArr) -> tuple[FloatArr, FloatArr]:
        lam, mu = np.asarray(lam, dtype=np.float64), np.asarray(mu, dtype=np.float64)
        s = np.log(lam) - np.log(mu)
        t = np.log(lam) + np.log(mu)
        s2 = self.a * s + self.b
        t2 = self.t_ref + self.d * (t - self.t_ref) + self.c
        return np.exp((t2 + s2) / 2), np.exp((t2 - s2) / 2)

    def adjust_grid(self, grid: FloatArr) -> FloatArr:
        return adjust_draws(grid, self.delta)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoalCalibration:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass(frozen=True)
class CardCalibration:
    d: float = 1.0  # pull of the expected total towards the league-average model
    f: float = 1.0  # scale of the negative-binomial dispersion k
    fitted_on: str = ""

    def mean(self, m: FloatArr, base: FloatArr) -> FloatArr:
        m, base = np.asarray(m, dtype=np.float64), np.asarray(base, dtype=np.float64)
        out: FloatArr = base ** (1 - self.d) * m**self.d
        return out

    def k(self, k: FloatArr) -> FloatArr:
        out: FloatArr = np.asarray(k, dtype=np.float64) * self.f
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CardCalibration:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def p_over_nb(m: FloatArr, k: FloatArr, line: float) -> FloatArr:
    """P(total > line) for negative-binomial totals with mean m and dispersion k (inf = Poisson)."""
    m, k = np.asarray(m, dtype=np.float64), np.asarray(k, dtype=np.float64)
    cut = int(np.floor(line))
    out = np.empty(len(m))
    inf = ~np.isfinite(k)
    out[inf] = 1 - poisson.cdf(cut, m[inf])
    fin = ~inf
    out[fin] = 1 - nbinom.cdf(cut, k[fin], k[fin] / (k[fin] + m[fin]))
    return out


# -- scoring ------------------------------------------------------------------------------
def _ll_binary(p: FloatArr, y: npt.NDArray[np.bool_]) -> float:
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(np.where(y, np.log(q), np.log(1 - q))))


def goal_markets(grids: FloatArr) -> dict[str, FloatArr]:
    """Match result, over 2.5 and both-score probabilities from many grids."""
    n = grids.shape[-1]
    gh, ga = np.indices((n, n))
    return {
        "p_h": (grids * (gh > ga)).sum(axis=(1, 2)),
        "p_d": (grids * (gh == ga)).sum(axis=(1, 2)),
        "p_a": (grids * (gh < ga)).sum(axis=(1, 2)),
        "p_over25": (grids * (gh + ga >= 3)).sum(axis=(1, 2)),
        "p_btts": (grids * ((gh > 0) & (ga > 0))).sum(axis=(1, 2)),
    }


def goal_scores(
    grids: FloatArr, home: npt.NDArray[np.intp], away: npt.NDArray[np.intp]
) -> dict[str, float]:
    """Log loss of match result, over/under 2.5, both-score and correct score."""
    mk = goal_markets(grids)
    res = np.where(home > away, 0, np.where(home == away, 1, 2))
    p3 = np.c_[mk["p_h"], mk["p_d"], mk["p_a"]]
    n = grids.shape[-1] - 1
    cell = grids[np.arange(len(home)), np.minimum(home, n), np.minimum(away, n)]
    return {
        "1x2": float(-np.mean(np.log(np.clip(p3[np.arange(len(res)), res], EPS, 1)))),
        "ou25": _ll_binary(mk["p_over25"], home + away >= 3),
        "btts": _ll_binary(mk["p_btts"], (home > 0) & (away > 0)),
        "score": float(-np.mean(np.log(np.clip(cell, EPS, 1)))),
    }


# -- fitting ------------------------------------------------------------------------------
def _grids_for(cal: GoalCalibration, lam: FloatArr, mu: FloatArr, rho: float) -> FloatArr:
    from football_edge.models.dixon_coles import score_grids

    lam2, mu2 = cal.expected(lam, mu)
    return adjust_draws(score_grids(lam2, mu2, rho), cal.delta)


def fit_goals(
    lam: FloatArr,
    mu: FloatArr,
    home: npt.NDArray[np.intp],
    away: npt.NDArray[np.intp],
    rho: float,
    label: str = "",
) -> GoalCalibration:
    """The five numbers that minimise match result + over/under + both-score log loss."""
    t_ref = float(np.mean(np.log(lam) + np.log(mu)))

    def make(th: FloatArr) -> GoalCalibration:
        return GoalCalibration(
            a=float(th[0]),
            b=float(th[1]),
            d=float(th[2]),
            c=float(th[3]),
            delta=float(th[4]),
            t_ref=t_ref,
        )

    def objective(th: FloatArr) -> float:
        if not (0.3 < th[0] < 3 and 0.05 < th[2] < 3 and abs(th[4]) < 1):
            return 1e9
        sc = goal_scores(_grids_for(make(th), lam, mu, rho), home, away)
        return sc["1x2"] + sc["ou25"] + sc["btts"]

    res = minimize(
        objective,
        np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 3000},
    )
    cal = make(res.x)
    return GoalCalibration(**{**cal.to_dict(), "fitted_on": label})


def fit_cards(
    m: FloatArr, base: FloatArr, k: FloatArr, total: FloatArr, label: str = ""
) -> CardCalibration:
    """The two numbers that minimise the summed log loss of the four card lines."""

    def objective(th: FloatArr) -> float:
        if not (0.05 < th[0] < 2 and 0.1 < th[1] < 20):
            return 1e9
        cal = CardCalibration(float(th[0]), float(th[1]))
        mm, kk = cal.mean(m, base), cal.k(k)
        return sum(_ll_binary(p_over_nb(mm, kk, x), total > x) for x in CARD_LINES)

    res = minimize(
        objective,
        np.array([1.0, 1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-7},
    )
    return CardCalibration(float(res.x[0]), float(res.x[1]), label)


def goal_losses(
    grids: FloatArr, home: npt.NDArray[np.intp], away: npt.NDArray[np.intp]
) -> FloatArr:
    """Per match: match result + over/under 2.5 + both-score log loss (for bootstraps)."""
    mk = goal_markets(grids)
    res = np.where(home > away, 0, np.where(home == away, 1, 2))
    p3 = np.c_[mk["p_h"], mk["p_d"], mk["p_a"]][np.arange(len(res)), res]
    ov = np.clip(mk["p_over25"], EPS, 1 - EPS)
    bt = np.clip(mk["p_btts"], EPS, 1 - EPS)
    o, b = home + away >= 3, (home > 0) & (away > 0)
    out: FloatArr = (
        -np.log(np.clip(p3, EPS, 1))
        - np.where(o, np.log(ov), np.log(1 - ov))
        - np.where(b, np.log(bt), np.log(1 - bt))
    )
    return out


def card_losses(m: FloatArr, k: FloatArr, total: FloatArr) -> FloatArr:
    """Per match: summed log loss of the four card lines."""
    out = np.zeros(len(m))
    for x in CARD_LINES:
        q = np.clip(p_over_nb(m, k, x), EPS, 1 - EPS)
        out -= np.where(total > x, np.log(q), np.log(1 - q))
    return out


def card_scores(m: FloatArr, k: FloatArr, total: FloatArr) -> dict[str, float]:
    return {f"cards_{x}": _ll_binary(p_over_nb(m, k, x), total > x) for x in CARD_LINES}


# -- walk-forward ---------------------------------------------------------------------------
def _span(seasons: pd.Series) -> str:
    return f"{int(seasons.min())}-{int(seasons.max())}, {len(seasons):,} matches"


def calibrate_goals_walk_forward(
    preds: pd.DataFrame, rho: float, first: int = 2019
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated walk-forward predictions: each season with a map fitted only on the seasons
    before it. Returns (predictions with calibrated columns and *_raw copies, one row per
    season with the map and raw/calibrated log losses)."""
    p = preds.dropna(subset=["exp_home", "exp_away"]).reset_index(drop=True)
    h, a = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    lam, mu = p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float)
    out, report = [], []
    for s in sorted(int(x) for x in p["season"].unique()):
        if s < first:
            continue
        tr = (p["season"] < s).to_numpy()
        te = (p["season"] == s).to_numpy()
        if not tr.any():
            continue
        cal = fit_goals(lam[tr], mu[tr], h[tr], a[tr], rho, _span(p.loc[tr, "season"]))
        out.append(apply_goals(p[te], cal, rho))
        raw = goal_scores(_grids_for(GoalCalibration(), lam[te], mu[te], rho), h[te], a[te])
        new = goal_scores(_grids_for(cal, lam[te], mu[te], rho), h[te], a[te])
        report.append(
            {
                "season": s,
                "n": int(te.sum()),
                **cal.to_dict(),
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"cal_{k}": v for k, v in new.items()},
            }
        )
    return pd.concat(out, ignore_index=True), pd.DataFrame(report)


def apply_goals(preds: pd.DataFrame, cal: GoalCalibration, rho: float) -> pd.DataFrame:
    """Calibrated copy of walk-forward predictions (raw columns kept with a _raw suffix)."""
    q = preds.copy()
    for col in ("p_h", "p_d", "p_a", "p_over25", "p_btts", "exp_home", "exp_away"):
        q[f"{col}_raw"] = q[col]
    lam2, mu2 = cal.expected(q["exp_home"].to_numpy(float), q["exp_away"].to_numpy(float))
    from football_edge.models.dixon_coles import score_grids

    mk = goal_markets(adjust_draws(score_grids(lam2, mu2, rho), cal.delta))
    for col, v in mk.items():
        q[col] = v
    q["exp_home"], q["exp_away"] = lam2, mu2
    q["cal_delta"] = cal.delta
    return q


def calibrate_cards_walk_forward(
    preds: pd.DataFrame, first: int = 2019
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same for the cards model (rows of evaluation/counts predictions, stat 'cards',
    with exp_team, exp_base and k_team)."""
    c = preds[(preds["stat"] == "cards") & preds["total"].notna()].reset_index(drop=True)
    m, base = c["exp_team"].to_numpy(float), c["exp_base"].to_numpy(float)
    k, tot = c["k_team"].to_numpy(float), c["total"].to_numpy(float)
    out, report = [], []
    for s in sorted(int(x) for x in c["season"].unique()):
        if s < first:
            continue
        tr, te = (c["season"] < s).to_numpy(), (c["season"] == s).to_numpy()
        if not tr.any():
            continue
        cal = fit_cards(m[tr], base[tr], k[tr], tot[tr], _span(c.loc[tr, "season"]))
        out.append(apply_cards(c[te], cal))
        raw = card_scores(m[te], k[te], tot[te])
        new = card_scores(cal.mean(m[te], base[te]), cal.k(k[te]), tot[te])
        report.append(
            {
                "season": s,
                "n": int(te.sum()),
                **cal.to_dict(),
                "raw_sum": sum(raw.values()),
                "cal_sum": sum(new.values()),
                **{f"raw_{x}": v for x, v in raw.items()},
                **{f"cal_{x}": v for x, v in new.items()},
            }
        )
    return pd.concat(out, ignore_index=True), pd.DataFrame(report)


def apply_cards(rows: pd.DataFrame, cal: CardCalibration) -> pd.DataFrame:
    q = rows.copy()
    m2 = cal.mean(q["exp_team"].to_numpy(float), q["exp_base"].to_numpy(float))
    k2 = cal.k(q["k_team"].to_numpy(float))
    for col in [x for x in q.columns if x.startswith("p_team_")]:
        ok = q[col].notna().to_numpy()
        if not ok.any():  # lines of other stats (corners 8.5...) stay empty
            continue
        q[f"{col}_raw"] = q[col]
        line = float(col.rsplit("_", 1)[1])
        q.loc[ok, col] = p_over_nb(m2[ok], k2[ok], line)
    q["exp_team_raw"], q["exp_team"], q["k_team"] = q["exp_team"], m2, k2
    return q


# -- storage -------------------------------------------------------------------------------
def save(
    path: Path,
    goals: GoalCalibration | None,
    cards: CardCalibration | None,
    adopted: dict[str, bool],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "goals": goals.to_dict() if goals else None,
        "cards": cards.to_dict() if cards else None,
        "adopted": adopted,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(path: Path) -> tuple[GoalCalibration | None, CardCalibration | None]:
    """The adopted calibrations (None where not adopted or no file)."""
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    adopted = data.get("adopted", {})
    g = data.get("goals") if adopted.get("goals") else None
    c = data.get("cards") if adopted.get("cards") else None
    return (
        GoalCalibration.from_dict(g) if g else None,
        CardCalibration.from_dict(c) if c else None,
    )


def block_bootstrap(
    diff: FloatArr, blocks: npt.NDArray[Any], reps: int = 2000, seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """`level` (95%) interval of the mean of `diff` resampling whole blocks (league x week)."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    k = len(codes)
    means = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, k, k)
        means[i] = sums[pick].sum() / counts[pick].sum()
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)
