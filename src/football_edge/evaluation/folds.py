"""Chronological walk-forward folds.

Spec: docs/A1_1_fold_spec.md.  Tests: tests/unit/test_folds.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class Fold:
    snapshot: pd.Timestamp
    train_idx: npt.NDArray[np.intp]  # positional indices into the matches frame
    test_idx: npt.NDArray[np.intp]


def make_folds(
    matches: pd.DataFrame,
    first_test_season: int = 2018,
    last_test_season: int = 2025,
    min_train_matches: int = 1500,
) -> list[Fold]:
    """One fold per ISO week: train on all matches before Monday 00:00,
    test on that week's matches from the test seasons.

    Decisions:
    1. COVID restart: weeks with no matches never become snapshots. No week
       mixes two seasons (2019/20 ended 2 Aug 2020, 2020/21 began 12 Sep).
       The first post-restart fold trains on stale pre-March data; down-weighting
       it and the no-crowd home advantage are the model's job.
    2. Expanding window: all history is kept because more data stabilises the
       estimates; recency is handled by time decay inside the model, so a
       window cut would only discard information and add a tuning knob.
    3. Promoted teams: their test matches are kept, because dropping them would
       score the model only on easy, known teams (selection bias). Predicting
       teams with no history is the model's job (shrinkage), not the split's.
    """
    dates = matches["date"]
    week_start = dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")
    is_test_season = matches["season"].between(first_test_season, last_test_season)
    snapshots = sorted(week_start[is_test_season].unique())
    folds: list[Fold] = []
    for snap in snapshots:
        snapshot = pd.Timestamp(snap)
        week_end = snapshot + pd.Timedelta(days=7)
        in_train = dates < snapshot
        in_test = is_test_season & (dates >= snapshot) & (dates < week_end)
        train_idx = np.flatnonzero(in_train.to_numpy())
        test_idx = np.flatnonzero(in_test.to_numpy())
        if len(test_idx) == 0 or len(train_idx) < min_train_matches:
            continue
        folds.append(Fold(snapshot, train_idx, test_idx))
    return folds
