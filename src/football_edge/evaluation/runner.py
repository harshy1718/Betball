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

# Everything known only after kick-off; removed from test rows before predicting.
_RESULT_COLS = [
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]


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
        hidden = test.drop(columns=_RESULT_COLS, errors="ignore")
        probs = model.predict_proba(hidden)
        _check_probs(probs, len(test), model.name)
        extra_fn = getattr(model, "predict_extra", None)
        extra = extra_fn(hidden) if extra_fn is not None else {}
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
                    # actual goals, for scoring goals markets (never shown to the model)
                    "goals_home": test["FTHG"].to_numpy(),
                    "goals_away": test["FTAG"].to_numpy(),
                    **extra,
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
        row = {"n": len(g), "log_loss": log_loss(p, y), "brier": brier_score(p, y)}
        goals = g["goals_home"] + g["goals_away"]
        if "p_over25" in g:  # goals markets, when the model provides them
            over = g["p_over25"].to_numpy()
            row["ou25_log_loss"] = log_loss(
                np.c_[over, 1 - over], np.asarray(goals < 3, dtype=np.intp)
            )
        if "p_btts" in g:
            btts = g["p_btts"].to_numpy()
            yes = ((g["goals_home"] > 0) & (g["goals_away"] > 0)).to_numpy()
            row["btts_log_loss"] = log_loss(np.c_[btts, 1 - btts], (~yes).astype(np.intp))
        return pd.Series(row)

    rows = [score(preds).rename("all")]
    for col in ("league", "season"):
        for key, g in preds.groupby(col):
            rows.append(score(g).rename(f"{col}={key}"))
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    return out
