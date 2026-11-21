import numpy as np
import pandas as pd

from football_edge.markets.price_edge import VARIANTS, run_variant, select_bets

PRE_2 = next(v for v in VARIANTS if v.name == "pre_2")


def _frame(rows):
    cols = ["pre_h", "pre_d", "pre_a", "max_h", "max_d", "max_a", "FTR"]
    df = pd.DataFrame(rows, columns=cols)
    n = len(df)
    df["league"] = "E0"
    df["season"] = 2020
    df["date"] = pd.Timestamp("2020-09-12") + pd.to_timedelta(np.arange(n) * 7, unit="D")
    df["home"], df["away"] = "A", "B"
    df["pre_src"] = "PS"
    df["close_src"] = "PS"
    df[["close_h", "close_d", "close_a"]] = df[["pre_h", "pre_d", "pre_a"]].to_numpy()
    return df


def test_bets_only_when_best_price_beats_fair_by_threshold():
    # Pinnacle 2.0/4.0/4.0 has no margin: fair = 0.5/0.25/0.25.
    rows = [
        [2.0, 4.0, 4.0, 2.10, 4.0, 4.0, "H"],  # home edge 5%: bet home
        [2.0, 4.0, 4.0, 2.02, 4.0, 4.0, "H"],  # home edge 1%: no bet
        [2.0, 4.0, 4.0, 2.0, 4.0, 4.4, "D"],  # away edge 10%: bet away, loses
    ]
    bets = select_bets(_frame(rows), PRE_2)
    assert bets["pick"].tolist() == ["H", "A"]
    np.testing.assert_allclose(bets["edge"], [0.05, 0.10])
    np.testing.assert_allclose(bets["profit"], [110.0, -100.0])


def test_non_pinnacle_reference_is_skipped():
    df = _frame([[2.0, 4.0, 4.0, 2.2, 4.0, 4.0, "H"]])
    df["pre_src"] = "B365"
    assert select_bets(df, PRE_2).empty


def test_closing_line_value_measured_against_close():
    df = _frame([[2.0, 4.0, 4.0, 2.2, 4.0, 4.0, "H"]])
    df[["close_h", "close_d", "close_a"]] = [1.8, 4.5, 4.5]  # market moved towards home
    bets = select_bets(df, PRE_2)
    fair_close_home = (1 / 1.8) / (1 / 1.8 + 2 / 4.5)
    assert bets["clv"].iloc[0] == 2.2 * fair_close_home - 1


def test_variant_fails_with_too_few_bets():
    res, _ = run_variant(
        _frame([[2.0, 4.0, 4.0, 2.2, 4.0, 4.0, "H"]]), PRE_2, level=0.95, n_boot=50
    )
    assert not res.criteria["at least 200 bets"]
    assert not res.passed
