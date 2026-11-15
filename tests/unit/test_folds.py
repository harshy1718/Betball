import numpy as np
import pandas as pd
import pytest

from football_edge.evaluation.folds import make_folds

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def matches(cfg):
    from football_edge.data.loader import load_matches

    return load_matches(cfg)


@pytest.fixture(scope="module")
def folds(matches):
    return make_folds(matches)


def test_train_strictly_before_snapshot(matches, folds):
    for f in folds:
        assert matches.loc[f.train_idx, "date"].max() < f.snapshot


def test_test_window_is_one_week_from_snapshot(matches, folds):
    for f in folds:
        d = matches.loc[f.test_idx, "date"]
        assert d.min() >= f.snapshot and d.max() < f.snapshot + pd.Timedelta(days=7)


def test_no_train_test_overlap(folds):
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)


def test_every_test_match_in_exactly_one_fold(matches, folds):
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(all_test) == len(set(all_test))
    s = matches.loc[all_test, "season"]
    assert s.min() >= 2018 and s.max() <= 2025


def test_folds_are_chronological(folds):
    snaps = [f.snapshot for f in folds]
    assert snaps == sorted(snaps)


def test_input_not_mutated(matches):
    before = matches.copy()
    make_folds(matches)
    pd.testing.assert_frame_equal(matches, before)


def test_expected_fold_counts(matches, folds):
    # Guards against an empty or partial result. Without this, the tests above
    # pass on an empty list because their loops never run.
    assert len(folds) == 294
    assert sum(len(f.test_idx) for f in folds) == 3 * 8 * 380
    assert len(folds[0].train_idx) == 3420 and len(folds[0].test_idx) == 10
