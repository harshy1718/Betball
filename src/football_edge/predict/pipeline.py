"""One call from fixtures to the full forecast table (goals, market view, corners, cards)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_edge.config import Config
from football_edge.predict.fixtures import load_fixtures
from football_edge.predict.forecast import (
    TrainingInfo,
    count_forecasts,
    fit_for,
    forecast,
    market_view,
)


def forecast_fixtures(
    cfg: Config, history: pd.DataFrame, fixtures: pd.DataFrame
) -> tuple[pd.DataFrame, TrainingInfo]:
    model_name = cfg.predict.get("model", "dixon_coles")
    model, info = fit_for(history, fixtures, cfg.models.get(model_name, {}))
    fc = market_view(forecast(model, fixtures))
    return pd.concat([fc, count_forecasts(history, fixtures)], axis=1), info


def attach_prices(
    cfg: Config, fixtures: pd.DataFrame, known: dict[str, set[str]], aliases: Path
) -> pd.DataFrame:
    """Fill odds_/ref_ columns from data/upcoming/fixtures.csv where it lists the match."""
    path = cfg.root / "data" / "upcoming" / "fixtures.csv"
    if not path.exists():
        return fixtures
    priced = load_fixtures(path, known, cfg.leagues, cfg.predict.get("odds_source", "Avg"),
                           aliases, allow_unknown=True)  # fmt: skip
    keys = ["league", "date", "home", "away"]
    cols = [c for c in priced.columns if c.startswith(("odds_", "ref_"))]
    base = fixtures.drop(columns=[c for c in cols if c in fixtures.columns])
    return base.merge(priced[keys + cols], on=keys, how="left")
