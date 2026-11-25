"""Fit Dixon-Coles on all history before the fixtures and forecast every market.

This is the same model, with the same locked settings, that was evaluated week by week
in reports/evaluation/dixon_coles. Nothing is re-tuned here: predicting live must use
exactly the procedure whose track record we measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_edge.config import Config
from football_edge.models.calibration import CardCalibration, GoalCalibration
from football_edge.models.dixon_coles import DixonColes, DixonColesParams

log = logging.getLogger(__name__)

# Market -> (selection names, probability columns), in a fixed order.
MARKETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "1x2": (("home", "draw", "away"), ("p_h", "p_d", "p_a")),
    "ou25": (("over", "under"), ("p_over25", "p_under25")),
    "btts": (("yes", "no"), ("p_btts_yes", "p_btts_no")),
}
# Matching price column for each probability column.
PRICE_OF = {
    "p_h": "odds_h",
    "p_d": "odds_d",
    "p_a": "odds_a",
    "p_over25": "odds_over25",
    "p_under25": "odds_under25",
    "p_btts_yes": "odds_btts_yes",
    "p_btts_no": "odds_btts_no",
}


@dataclass(frozen=True)
class TrainingInfo:
    n_matches: int
    last_match: pd.Timestamp
    converged: bool


def fit_for(
    history: pd.DataFrame, fixtures: pd.DataFrame, params: dict[str, object]
) -> tuple[DixonColes, TrainingInfo]:
    """Fit on every match played before the earliest fixture (nothing from the future)."""
    cutoff = fixtures["date"].min()
    train = history[history["date"] < cutoff]
    if train.empty:
        raise ValueError("no history before the first fixture")
    model = DixonColes(DixonColesParams.from_dict(dict(params)))
    model.fit(train)
    info = TrainingInfo(len(train), pd.Timestamp(train["date"].max()), model.converged)
    if not info.converged:
        log.warning("optimiser did not fully converge; treat these numbers with care")
    return model, info


def load_calibration(cfg: Config) -> tuple[GoalCalibration | None, CardCalibration | None]:
    """The adopted calibration maps (ADR 0020), or (None, None) when switched off."""
    from football_edge.models.calibration import load

    c = cfg.calibration
    if not c.get("enabled", False):
        return None, None
    return load(cfg.root / str(c.get("path", "reports/calibration/calibration.json")))


def forecast(model: DixonColes, fixtures: pd.DataFrame, top_scores: int = 3) -> pd.DataFrame:
    """One row per fixture: all market probabilities, expected goals, likeliest scores.
    With a calibration attached to the model, every number is calibrated and the raw
    expected goals are kept in raw_exp_home / raw_exp_away (for the prediction log)."""
    lam, mu = model.expected_goals(fixtures)
    raw_lam, raw_mu = model.raw_expected_goals(fixtures)
    g = np.arange(model.params.max_goals + 1)
    total = g[:, None] + g[None, :]
    rows = []
    calibrated = getattr(model, "calibration", None) is not None
    for k in range(len(fixtures)):
        m = model.score_matrix(float(lam[k]), float(mu[k]))
        # the uncalibrated grid: for the log's raw chances and for the golden bars, which
        # were searched on raw chances (betting/golden.py: unadjusted)
        r = model.raw_score_matrix(float(raw_lam[k]), float(raw_mu[k])) if calibrated else m
        r_top = int(np.argmax(r))
        order = np.argsort(m, axis=None)[::-1][:top_scores]
        scores = [
            f"{i}-{j} ({m[i, j]:.0%})"
            for i, j in zip(*np.unravel_index(order, m.shape), strict=True)
        ]
        p_over = float(m[total >= 3].sum())
        p_btts = float(m[1:, 1:].sum())
        rows.append(
            {
                "p_h": float(np.tril(m, -1).sum()),
                "p_d": float(np.trace(m)),
                "p_a": float(np.triu(m, 1).sum()),
                "p_over25": p_over,
                "p_under25": 1.0 - p_over,
                "p_btts_yes": p_btts,
                "p_btts_no": 1.0 - p_btts,
                "exp_home": float(lam[k]),
                "exp_away": float(mu[k]),
                "raw_exp_home": float(raw_lam[k]),
                "raw_exp_away": float(raw_mu[k]),
                "raw_p_h": float(np.tril(r, -1).sum()),
                "raw_p_d": float(np.trace(r)),
                "raw_p_a": float(np.triu(r, 1).sum()),
                "raw_p_over25": float(r[total >= 3].sum()),
                "raw_top_score": f"{r_top // r.shape[1]}-{r_top % r.shape[1]}",
                "raw_top_score_p": float(r.flat[r_top]),
                "top_scores": ", ".join(scores),
                "top_score": scores[0].split(" ")[0],
                "top_score_p": float(m.flat[order[0]]),
            }
        )
    return pd.concat([fixtures.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def market_view(fc: pd.DataFrame) -> pd.DataFrame:
    """Add the market's fair probabilities: mkt_* columns plus mkt_src_<market>.

    Per fixture and market, the reference prices (ref_*, Pinnacle) are used when all of
    them are present, else the site's own prices (odds_*), else none. The margin is removed
    proportionally (markets/margin.py). On 2020-25, margin-free Pinnacle prices scored
    0.967 log-loss before kick-off and 0.964 at the close, against 0.981 for our best model
    (docs/adr/0007), so this is the more accurate probability whenever it exists.
    """
    from football_edge.markets.margin import remove_margin

    out = fc.copy()
    for market, (_, pcols) in MARKETS.items():
        mcols = ["mkt_" + c[2:] for c in pcols]
        out[mcols] = np.nan
        out[f"mkt_src_{market}"] = "-"
        for prefix, label in (("ref_", "sharp"), ("odds_", "site")):
            cols = [PRICE_OF[c].replace("odds_", prefix) for c in pcols]
            if not all(c in out.columns for c in cols):
                continue
            prices = out[cols].to_numpy(dtype=float)
            ok = ~np.isnan(prices).any(axis=1) & (out[f"mkt_src_{market}"] == "-").to_numpy()
            if ok.any():
                out.loc[ok, mcols] = remove_margin(prices[ok])
                out.loc[ok, f"mkt_src_{market}"] = label
    return out


LIVE_STATS = ("cards", "yellows", "reds")  # corners dropped: no skill over the league average


def count_forecasts(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    stats: tuple[str, ...] = LIVE_STATS,
    cards_cal: CardCalibration | None = None,
) -> pd.DataFrame:
    """Cards, yellow cards and red cards for each fixture (models/counts.py).

    Trained on every match before the earliest fixture, like the goals model. Columns:
    exp_<stat> (expected match total) and p_<stat>_over_<line> for the standard lines. Corners
    are no longer forecast: the team model never beat the league average (ADR 0008, 0020).
    With `cards_cal` the cards numbers are calibrated (ADR 0020) and the raw chances kept in
    raw_p_cards_over_<line>.
    """
    from football_edge.models.calibration import p_over_nb
    from football_edge.models.counts import STATS, CountParams, TeamCountModel, with_card_totals

    train = with_card_totals(history[history["date"] < fixtures["date"].min()])
    out = pd.DataFrame(index=range(len(fixtures)))
    fx = fixtures.reset_index(drop=True)
    for stat in stats:
        hcol, acol, lines = STATS[stat]
        if not {hcol, acol} <= set(train.columns):
            continue
        # Red cards: the league-average model; team effects scored worse (ADR 0008).
        m = TeamCountModel(stat, CountParams(team_effects=stat != "reds"))
        m.fit(train)
        lam, mu = m.expected(fx)
        exp = lam + mu
        if stat == "cards" and cards_cal is not None:
            b = TeamCountModel(stat, CountParams(team_effects=False))
            b.fit(train)
            bl, bm = b.expected(fx)
            k = np.array([m.k.get(lg, float("inf")) for lg in fx["league"]])
            exp2, k2 = cards_cal.mean(exp, bl + bm), cards_cal.k(k)
            out[f"exp_{stat}"] = exp2
            for line in lines:
                out[f"p_{stat}_over_{line}"] = p_over_nb(exp2, k2, line)
                out[f"raw_p_{stat}_over_{line}"] = m.p_over(fx, line)
            continue
        out[f"exp_{stat}"] = exp
        for line in lines:
            out[f"p_{stat}_over_{line}"] = m.p_over(fx, line)
    return out
