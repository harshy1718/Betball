"""Walk-forward evaluation: fit a fresh model per fold, predict that fold's week."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.metrics import brier_score, log_loss
from football_edge.models.base import Model, outcome_index

log = logging.getLogger(__name__)

_RESULT_COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG"]


def walk_forward(
    matches: pd.DataFrame, folds: list[Fold], make_model: Callable[[], Model]
) -> pd.DataFrame:
    """Return one row per test match with the model's H/D/A probabilities."""
    parts = []
    for n, fold in enumerate(folds, start=1):
        model = make_model()
        model.fit(matches.iloc[fold.train_idx])
        test = matches.iloc[fold.test_idx]
        # Remove results before predicting, so a model physically cannot peek.
        probs = model.predict_proba(test.drop(columns=_RESULT_COLS, errors="ignore"))
        _check_probs(probs, len(test), model.name)
        parts.append(
            pd.DataFrame(
                {
                    "row": fold.test_idx,
                    "snapshot": fold.snapshot,
                    "league": test["league"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "home": test["home"].to_numpy(),
                    "away": test["away"].to_numpy(),
                    "outcome": outcome_index(test["FTR"]),
                    "p_h": probs[:, 0],
                    "p_d": probs[:, 1],
                    "p_a": probs[:, 2],
                }
            )
        )
        if n % 50 == 0:
            log.info("%s: %d/%d folds", model.name, n, len(folds))
    return pd.concat(parts, ignore_index=True)


def _check_probs(probs: np.ndarray, n: int, name: str) -> None:
    if probs.shape != (n, 3):
        raise ValueError(f"{name}: expected shape {(n, 3)}, got {probs.shape}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6) or (probs < 0).any():
        raise ValueError(f"{name}: probabilities must be non-negative and sum to 1")


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    """Scores overall, by league and by season (lower is better for both)."""

    def score(g: pd.DataFrame) -> pd.Series:
        p = g[["p_h", "p_d", "p_a"]].to_numpy()
        y = g["outcome"].to_numpy()
        return pd.Series({"n": len(g), "log_loss": log_loss(p, y), "brier": brier_score(p, y)})

    rows = [score(preds).rename("all")]
    for col in ("league", "season"):
        for key, g in preds.groupby(col):
            rows.append(score(g).rename(f"{col}={key}"))
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    return out
