import numpy as np
import pandas as pd
import pytest

from football_edge.betting.ledger import (
    month_status,
    new_entries,
    read_log,
    settle,
    summary,
    write_log,
)

MATCHES = pd.DataFrame(
    {
        "league": ["E0", "E0"],
        "date": pd.to_datetime(["2026-11-22", "2026-11-22"]),
        "home": ["A", "C"],
        "away": ["B", "D"],
        "FTHG": [2, 0],
        "FTAG": [1, 0],
    }
)


def _singles():
    return pd.DataFrame(
        {
            "league": ["E0", "E0", "E0"],
            "date": pd.to_datetime(["2026-11-22"] * 3),
            "home": ["A", "C", "E"],
            "away": ["B", "D", "F"],
            "market": ["1x2", "btts", "ou25"],
            "selection": ["home", "yes", "over"],
            "odds": [1.8, 2.0, np.nan],
            "stake": [200.0, 100.0, 100.0],
            "prob": [0.6, 0.5, 0.6],
        }
    )


def test_log_roundtrip_settle_and_summary(tmp_path):
    doubles = pd.DataFrame(
        {
            "leg1": ["E0|2026-11-22|A|B|1x2|home"],
            "leg2": ["E0|2026-11-22|C|D|ou25|under"],
            "odds": [3.0],
            "stake": [100.0],
            "prob": [0.36],
        }
    )
    log = new_entries(_singles(), doubles, pd.Timestamp("2026-11-20"))
    assert len(log) == 3  # the unpriced single cannot be logged
    path = tmp_path / "log.csv"
    write_log(log, path)
    log = read_log(path)
    assert month_status(log, pd.Timestamp("2026-11-25")).pending_stake == 400

    log, n = settle(log, MATCHES)
    assert n == 3
    assert log["result"].tolist() == ["win", "loss", "win"]  # 2-1 home; 0-0 no btts; both legs won
    assert log["profit"].tolist() == pytest.approx([160.0, -100.0, 200.0])
    status = month_status(log, pd.Timestamp("2026-11-25"))
    assert (status.settled_profit, status.pending_stake) == (260.0, 0.0)
    s = summary(log)
    assert s.loc["2026-11", "bets"] == 3 and s.loc["2026-11", "wins_expected"] == pytest.approx(
        1.46
    )


def test_unplayed_matches_stay_pending():
    log = new_entries(
        _singles().iloc[:1].assign(home="X"),
        pd.DataFrame(columns=["leg1", "leg2", "odds", "stake", "prob"]),
        pd.Timestamp("2026-11-20"),
    )
    log, n = settle(log, MATCHES)
    assert n == 0 and log["result"].isna().all()
