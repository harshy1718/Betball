import numpy as np
import pandas as pd
import pytest

from football_edge.betting.simulate import (
    GD_RANGE,
    ah_halves,
    ah_return,
    ah_win_loss,
    simulate,
    with_margin,
)
from football_edge.betting.staking import StakingPolicy

POLICY = {
    "monthly_budget": 10000,
    "currency": "INR",
    "tiers": {
        "1x2": [[0.50, 0.01], [0.60, 0.02], [0.70, 0.03]],
        "ou25": [[0.58, 0.01], [0.65, 0.02], [0.72, 0.03]],
        "btts": [[0.58, 0.01], [0.65, 0.02], [0.72, 0.03]],
    },
    "price_tolerance": 0.06,
    "max_match_fraction": 0.04,
    "stop_loss_fraction": 0.30,
    "stake_rounding": 10,
    "doubles": {"max_count": 2, "stake_fraction": 0.01, "min_leg_tier": "high"},
}


@pytest.mark.parametrize(
    ("line", "gd", "odds", "expected"),
    [
        (-0.5, 1, 1.9, 0.9),  # home -0.5, wins by 1: full win
        (-0.5, 0, 1.9, -1.0),  # draw: loss
        (-1.0, 1, 1.9, 0.0),  # home -1, wins by 1: stake back
        (-0.75, 1, 1.9, 0.45),  # half win (-0.5), half push (-1)
        (-0.75, 0, 1.9, -1.0),
        (-0.25, 0, 1.9, -0.5),  # half push (0), half loss (-0.5)
        (0.25, 0, 1.9, 0.45),  # half push (0), half win (+0.5)
        (1.5, -1, 2.0, 1.0),  # home +1.5 loses by 1: covers
    ],
)
def test_asian_handicap_settlement(line, gd, odds, expected):
    margins = tuple(gd + h for h in ah_halves(line))
    assert ah_return(margins, odds) == pytest.approx(expected)


def test_ah_probabilities_from_goal_difference():
    gd = np.zeros(len(GD_RANGE))
    ks = list(GD_RANGE)
    gd[ks.index(0)], gd[ks.index(1)], gd[ks.index(2)] = 0.3, 0.4, 0.3
    w, lo = ah_win_loss(gd, -0.75, "home")  # by 2: win; by 1: half win; draw: loss
    assert (w, lo) == pytest.approx((0.3 + 0.5 * 0.4, 0.3))
    w, lo = ah_win_loss(gd, -0.75, "away")
    assert (w, lo) == pytest.approx((0.3, 0.3 + 0.5 * 0.4))


def test_with_margin_sets_overround():
    out = with_margin(np.array([[2.0, 2.0]]), 0.08)
    assert (1 / out).sum() == pytest.approx(1.08)


def _cands(ret, n_weeks=4):
    rows = []
    for w in range(n_weeks):
        rows.append(
            {
                "week": pd.Timestamp("2021-01-04") + pd.Timedelta(weeks=w),
                "row": w,
                "date": pd.Timestamp("2021-01-05") + pd.Timedelta(weeks=w),
                "season": 2020,
                "market": "1x2",
                "pick": "home",
                "w": 0.75,
                "l": 0.25,
                "conf": 0.75,
                "odds": 1.5,
                "ev": 0.75 * 1.5 - 1,
                "ret": ret,
            }
        )
    return pd.DataFrame(rows)


def test_balance_arithmetic_and_stakes_follow_balance():
    pol = StakingPolicy.from_dict(POLICY)
    res = simulate(_cands(0.5, n_weeks=12), pol, 10000)  # every bet wins at 1.5
    # high tier = 3% of balance, capped at 4% per match; rounded down to 10
    expected = 10000.0
    for _ in range(12):
        stake = np.floor(0.03 * expected / 10) * 10
        expected += stake * 0.5
    assert res.final == pytest.approx(expected)
    assert res.bets["stake"].iloc[-1] > res.bets["stake"].iloc[0]


def test_losing_bets_shrink_balance():
    pol = StakingPolicy.from_dict(POLICY)
    res = simulate(_cands(-1.0), pol, 10000)
    assert res.final < 10000 and res.lowest == res.final
