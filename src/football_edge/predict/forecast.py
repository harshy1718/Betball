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


def forecast(model: DixonColes, fixtures: pd.DataFrame, top_scores: int = 3) -> pd.DataFrame:
    """One row per fixture: all market probabilities, expected goals, likeliest scores."""
    lam, mu = model.expected_goals(fixtures)
    g = np.arange(model.params.max_goals + 1)
    total = g[:, None] + g[None, :]
    rows = []
    for k in range(len(fixtures)):
        m = model.score_matrix(float(lam[k]), float(mu[k]))
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
                "top_scores": ", ".join(scores),
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


def count_forecasts(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Corners, yellow cards and red cards for each fixture (models/counts.py).

    Trained on every match before the earliest fixture, like the goals model. Columns:
    exp_<stat> (expected match total) and p_<stat>_over_<line> for the standard lines.
    """
    from football_edge.models.counts import STATS, CountParams, TeamCountModel, with_card_totals

    train = with_card_totals(history[history["date"] < fixtures["date"].min()])
    out = pd.DataFrame(index=range(len(fixtures)))
    fx = fixtures.reset_index(drop=True)
    for stat, (hcol, acol, lines) in STATS.items():
        if not {hcol, acol} <= set(train.columns):
            continue
        # Red cards: the league-average model; team effects scored worse (ADR 0008).
        m = TeamCountModel(stat, CountParams(team_effects=stat != "reds"))
        m.fit(train)
        lam, mu = m.expected(fx)
        out[f"exp_{stat}"] = lam + mu
        for line in lines:
            out[f"p_{stat}_over_{line}"] = m.p_over(fx, line)
    return out
