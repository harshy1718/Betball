"""Tuning and calibration per league (ADR 0023)."""

from __future__ import annotations

import itertools
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.metrics import log_loss
from football_edge.evaluation.runner import _RESULT_COLS
from football_edge.models import calibration as cal
from football_edge.models.base import outcome_index
from football_edge.models.dixon_coles import DixonColes, DixonColesParams, score_grids

_T: dict[str, Any] = {}


def _score(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    lg, params = task
    m, folds = _T["matches"], _T["folds"]
    probs, ys = [], []
    for f in folds:
        tr = m.iloc[f.train_idx]
        te = m.iloc[f.test_idx]
        tr, te = tr[tr["league"] == lg], te[te["league"] == lg]
        if te.empty:
            continue
        model = DixonColes(DixonColesParams.from_dict(params))
        model.fit(tr)
        probs.append(model.predict_proba(te.drop(columns=_RESULT_COLS, errors="ignore")))
        ys.append(outcome_index(te["FTR"]))
    p, y = np.vstack(probs), np.concatenate(ys)
    return {
        "league": lg,
        **{k: params[k] for k in _T["names"]},
        "log_loss": log_loss(p, y),
        "n": len(y),
    }


def tune(
    matches: pd.DataFrame,
    folds: list[Fold],
    base: dict[str, Any],
    grid: dict[str, list[Any]],
    seasons: tuple[int, int],
    workers: int = 2,
) -> pd.DataFrame:
    """Every grid combination for every league, scored on that league's tuning seasons only."""
    lo, hi = seasons
    tune_folds = [f for f in folds if matches["season"].iloc[f.test_idx].between(lo, hi).all()]
    names = list(grid)
    combos = [dict(zip(names, v, strict=True)) for v in itertools.product(*grid.values())]
    tasks = [(str(lg), {**base, **c}) for lg in sorted(matches["league"].unique()) for c in combos]
    state = {"matches": matches, "folds": tune_folds, "names": names}
    with Pool(workers, initializer=_T.update, initargs=(state,)) as pool:
        rows = pool.map(_score, tasks)
    return pd.DataFrame(rows).sort_values(["league", "log_loss"]).reset_index(drop=True)


def calibrate_by_league(preds: pd.DataFrame, rho: float, first: int = 2019) -> pd.DataFrame:
    """Walk-forward calibration with one map per league (each from its own earlier seasons)."""
    parts = []
    for _lg, g in preds.groupby("league"):
        out, _ = cal.calibrate_goals_walk_forward(g.reset_index(drop=True), rho, first)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def calibrated_losses(p: pd.DataFrame, rho: float) -> np.ndarray:
    """Per match: result + over/under + both-score log loss of calibrated predictions."""
    g = cal.adjust_draws(
        score_grids(p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float), rho),
        p["cal_delta"].to_numpy(float),
    )
    out: np.ndarray = cal.goal_losses(
        g, p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    )
    return out


def confirm_2026(
    hist: pd.DataFrame, p26: pd.DataFrame, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """2026/27 losses: one shared map vs one map per league, all fitted on 2018-2025."""

    def fit(d: pd.DataFrame) -> cal.GoalCalibration:
        d = d.dropna(subset=["exp_home", "exp_away"])
        return cal.fit_goals(
            d["exp_home"].to_numpy(float),
            d["exp_away"].to_numpy(float),
            d["goals_home"].to_numpy(int),
            d["goals_away"].to_numpy(int),
            rho,
        )

    shared = fit(hist)
    q = p26.dropna(subset=["exp_home", "exp_away"]).reset_index(drop=True)
    a = calibrated_losses(cal.apply_goals(q, shared, rho), rho)
    b = np.empty(len(q))
    for lg, g in q.groupby("league"):
        m = fit(hist[hist["league"] == lg])
        b[g.index.to_numpy()] = calibrated_losses(cal.apply_goals(g, m, rho), rho)
    return a, b
