import numpy as np
import pandas as pd

from football_edge.predict.confidence import build_track_record, lookup


def _goals(n=4000, seed=0, calibrated=True):
    """Fake walk-forward predictions whose outcomes follow (or ignore) the probabilities."""
    rng = np.random.default_rng(seed)
    p_h = rng.uniform(0.2, 0.7, n)
    p_d = np.full(n, 0.25)
    p_a = 1 - p_h - p_d
    u = rng.random(n)
    if not calibrated:
        u = rng.permutation(u)[np.argsort(np.argsort(-p_h))]  # outcomes unrelated to p_h
    gh = np.where(u < p_h, 2, np.where(u < p_h + p_d, 1, 0))
    ga = np.where(u < p_h, 0, np.where(u < p_h + p_d, 1, 2))
    return pd.DataFrame(
        {
            "row": np.arange(n),
            "season": 2021,
            "p_h": p_h,
            "p_d": p_d,
            "p_a": p_a,
            "p_over25": 0.5,
            "p_btts": 0.5,
            "goals_home": gh,
            "goals_away": ga,
        }
    )


def _matches(n):
    return pd.DataFrame(
        {c: np.nan for c in ["close_h", "close_d", "close_a", "ou_close_over", "ou_close_under"]},
        index=range(n),
    )


def test_calibrated_market_gets_high_confidence():
    g = _goals()
    table = build_track_record(g, None, _matches(len(g)))
    label, text = lookup(table, "model", "1x2", 0.62)
    assert label in ("High", "Medium") and "came true" in text
    assert set(table["label"]) <= {"High", "Medium", "Low"}


def test_small_bands_and_missing_records_are_low():
    g = _goals(n=300)
    table = build_track_record(g, None, _matches(len(g)))
    assert (table.loc[table["n"] < 200, "label"] == "Low").all()
    assert lookup(table, "model", "corners_9.5", 0.5)[0] == "Low"
    assert lookup(None, "model", "1x2", 0.5)[0] == "-"
