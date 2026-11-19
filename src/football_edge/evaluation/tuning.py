"""Grid search for model settings, judged only on early seasons.

Settings are chosen on the tuning seasons, then locked. Later seasons are never
looked at during the search, so the reported results on them stay honest.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.metrics import log_loss
from football_edge.evaluation.runner import walk_forward
from football_edge.models.registry import model_factory

log = logging.getLogger(__name__)


def grid_search(
    matches: pd.DataFrame,
    folds: list[Fold],
    model: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    tune_seasons: tuple[int, int],
) -> pd.DataFrame:
    """Score every combination in `grid` on folds whose test matches fall in `tune_seasons`."""
    lo, hi = tune_seasons
    tune_folds = [f for f in folds if matches["season"].iloc[f.test_idx].between(lo, hi).all()]
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = {**base_params, **dict(zip(names, values, strict=True))}
        preds = walk_forward(matches, tune_folds, model_factory(model, params))
        score = log_loss(preds[["p_h", "p_d", "p_a"]].to_numpy(), preds["outcome"].to_numpy())
        rows.append({**dict(zip(names, values, strict=True)), "log_loss": score, "n": len(preds)})
        log.info("%s -> %.5f", dict(zip(names, values, strict=True)), score)
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
