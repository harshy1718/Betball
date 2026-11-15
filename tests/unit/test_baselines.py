import numpy as np
import pandas as pd
import pytest

from football_edge.models.base import outcome_index
from football_edge.models.elo import Elo, EloParams
from football_edge.models.team_blind import TeamBlind


def _matches(rows):
    df = pd.DataFrame(rows, columns=["league", "season", "date", "home", "away", "FTR"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture()
def history():
    # A beats everyone; C loses to everyone. Two leagues to check separation.
    rows = []
    day = pd.Timestamp("2016-08-01")
    for i in range(30):
        d = day + pd.Timedelta(days=7 * i)
        season = 2015 if i < 10 else 2016
        rows += [
            ("E0", season, d, "A", "B", "H"),
            ("E0", season, d, "B", "C", "H"),
            ("E0", season, d, "C", "A", "A"),
            ("SP1", season, d, "X", "Y", "D"),
        ]
    return _matches(rows)


def test_outcome_index_maps_and_rejects():
    assert outcome_index(pd.Series(["H", "D", "A"])).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        outcome_index(pd.Series(["H", "X"]))


def test_team_blind_uses_league_rates(history):
    m = TeamBlind()
    m.fit(history)
    p = m.predict_proba(history.iloc[[0, 3]].drop(columns="FTR"))
    np.testing.assert_allclose(p[0], [2 / 3, 0, 1 / 3])  # E0: 2 home wins, 1 away win per round
    np.testing.assert_allclose(p[1], [0, 1, 0])  # SP1: all draws


def test_elo_ranks_strong_team_higher(history):
    m = Elo(EloParams(k=20))
    m.fit(history)
    assert m.ratings["E0"]["A"] > m.ratings["E0"]["B"] > m.ratings["E0"]["C"]
    test = _matches(
        [("E0", 2016, "2017-06-01", "A", "C", "H"), ("E0", 2016, "2017-06-01", "C", "A", "H")]
    )
    p = m.predict_proba(test.drop(columns="FTR"))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > p[1, 0]  # A at home to C is likelier a home win than C at home to A


def test_elo_prediction_does_not_change_ratings(history):
    m = Elo()
    m.fit(history)
    before = {lg: dict(t) for lg, t in m.ratings.items()}
    test = _matches([("E0", 2016, "2017-06-01", "A", "NEW", "H")])
    m.predict_proba(test.drop(columns="FTR"))
    assert m.ratings == before  # frozen at the snapshot; new team not stored


def test_elo_new_team_starts_near_bottom(history):
    m = Elo(EloParams(promoted_gap=0.0))
    m.fit(history)
    test = _matches([("E0", 2016, "2017-06-01", "NEW", "A", "H")])
    p_new = m.predict_proba(test.drop(columns="FTR"))
    test_c = _matches([("E0", 2016, "2017-06-01", "C", "A", "H")])
    p_c = m.predict_proba(test_c.drop(columns="FTR"))
    # Fewer than 3 active teams would mean initial rating; here 3 exist, so NEW ~ mean of A,B,C.
    assert p_new[0, 0] >= p_c[0, 0]
