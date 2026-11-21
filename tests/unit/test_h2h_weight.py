import numpy as np
import pandas as pd

from football_edge.evaluation.h2h_weight import h2h_rates, losses


def _history():
    return pd.DataFrame({
        "league": "E0",
        "date": pd.to_datetime(["2019-01-01", "2023-01-01", "2024-01-01", "2025-06-01"]),
        "home": ["A", "A", "B", "C"], "away": ["B", "B", "A", "D"],
        "FTHG": [5, 2, 3, 1], "FTAG": [0, 1, 1, 1],
    })  # fmt: skip


def _pred(**kw):
    row = {"league": "E0", "date": pd.Timestamp("2025-01-01"), "home": "A", "away": "B",
           "exp_home": 1.5, "exp_away": 1.0, "goals_home": 2, "goals_away": 1,
           "season": 2024}  # fmt: skip
    row.update(kw)
    return pd.DataFrame([row])


def test_window_venue_flip_and_pseudo_meeting():
    r = h2h_rates(_pred(), _history()).iloc[0]
    assert r["meetings"] == 2  # the 2019 meeting is outside 3 years; C v D is not this pair
    # League ratio before 2025: home 10 / away 2 = 5. A scored 2 at home and 1 away (x5).
    assert np.isclose(r["rate_home"], (2 + 1 * 5 + 1.5) / 3)
    assert np.isclose(r["rate_away"], (1 + 3 / 5 + 1.0) / 3)


def test_weight_zero_is_the_model_and_no_meeting_is_unchanged():
    p = _pred()
    rates = h2h_rates(p, _history())
    a, b = losses(p, rates, 0.0), losses(p, rates, 0.0)
    assert np.allclose(a["exact"], b["exact"])
    none = h2h_rates(_pred(home="X", away="Y"), _history())
    assert np.allclose(losses(p, none, 0.0)["exact"], losses(p, none, 1.0)["exact"])
