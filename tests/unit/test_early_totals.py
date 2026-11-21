import numpy as np
import pandas as pd

from football_edge.markets.early_totals import Variant, run_variant


def _matches(n=300, model_p=0.6, over_odds=2.0, over_rate=0.6):
    rng = np.random.default_rng(1)
    over = rng.random(n) < over_rate
    return pd.DataFrame({
        "league": "E0", "season": np.repeat([2020, 2021, 2022, 2023, 2024, 2025], n // 6),
        "date": pd.date_range("2020-08-01", periods=n, freq="3D"), "home": "A", "away": "B",
        "p_over25": model_p, "over": over, "b365_over": over_odds, "b365_under": 1.9,
        "fair_close": 0.55,
    })  # fmt: skip


def test_side_and_threshold():
    r = run_variant(_matches(), Variant("t", "b365", 0.03), level=0.9875, n_boot=50)
    assert len(r.bets) == 300 and set(r.bets["side"]) == {"over"}  # edge 0.6*2-1 = 20%
    r = run_variant(_matches(model_p=0.5), Variant("t", "b365", 0.03), level=0.9875, n_boot=50)
    assert r.bets.empty and not r.passed  # edges 0% and -5%: no bet


def test_a_real_edge_passes():
    r = run_variant(_matches(over_rate=0.62), Variant("t", "b365", 0.03), 0.9875, n_boot=200)
    assert r.roi > 0.15 and r.criteria["at least 200 bets"]
