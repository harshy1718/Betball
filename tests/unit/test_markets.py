"""ADR 0028 markets: definitions, half model, grading and slip maths."""

import json

import numpy as np
import pandas as pd
import pytest

from football_edge.models.dixon_coles import score_grid
from football_edge.tracking.results import leg_won
from football_edge.trader import markets as more
from football_edge.trader.desk import settle
from football_edge.trader.legs import Leg, goal_test
from football_edge.trader.slip import match_chance

GRID = score_grid(1.6, 1.1, -0.035)


def _m(market: str, sel: str, line: float | None = None) -> more.Market:
    mk = more.find(market, sel, line)
    assert mk is not None
    return mk


def test_every_market_is_found_and_keys_are_unique_per_selection() -> None:
    assert len(more.REGISTRY) == len(more.CATALOGUE)
    for mk in more.CATALOGUE:
        assert more.find(mk.market, mk.sel, mk.line) is mk


def test_joint_keeps_the_full_time_grid_and_half_shares() -> None:
    j = more.joint(GRID, 0.44, 0.45)
    assert np.allclose(j.sum(axis=(2, 3)), GRID)
    h, _a, h1, _a1 = np.indices(j.shape)
    assert (j * h1).sum() == pytest.approx(0.44 * (GRID.sum(1) * np.arange(11)).sum(), rel=1e-6)
    assert j[h1 > h].sum() == 0


def test_chances_add_up_and_refunds_are_conditional() -> None:
    cat = more.CATALOGUE
    p = dict(
        zip(
            [(m.market, m.sel, m.line) for m in cat],
            more.chances(GRID, cat, (np.array([0.44]), np.array([0.44])))[0],
            strict=True,
        )
    )
    exact = sum(p[("exact", s, None)] for s in ("0", "1", "2", "3", "4", "5", "6+"))
    assert exact == pytest.approx(1.0, abs=1e-9)
    htft = sum(p[("htft", f"{x}/{y}", None)] for x in "HDA" for y in "HDA")
    assert htft == pytest.approx(1.0, abs=1e-9)
    gh, ga = np.indices(GRID.shape)
    home, away = GRID[gh > ga].sum(), GRID[gh < ga].sum()
    assert p[("dnb", "home", None)] == pytest.approx(home / (home + away))
    assert p[("dnb", "home", None)] + p[("dnb", "away", None)] == pytest.approx(1.0)
    hsh = sum(p[("hsh", s, None)] for s in ("first", "second", "equal"))
    assert hsh == pytest.approx(1.0, abs=1e-9)


def test_grading_with_refunds_and_half_time() -> None:
    assert more.outcome(_m("dnb", "home"), 1, 1, 0, 0) is None  # draw: refunded
    assert more.outcome(_m("dnb", "home"), 2, 1, 0, 0) is True
    assert more.outcome(_m("ahc", "home-1"), 2, 1, 1, 0) is None  # won by exactly 1
    assert more.outcome(_m("ahc", "home-1"), 3, 1, 1, 0) is True
    assert more.outcome(_m("ahc", "away+1"), 2, 1, 1, 0) is None
    assert more.outcome(_m("ahc", "away+2"), 2, 1, 1, 0) is True
    assert more.outcome(_m("atot", "over", 3.0), 2, 1, 1, 0) is None
    assert more.outcome(_m("htft", "D/H"), 2, 1, 1, 1) is True
    assert more.outcome(_m("hsh", "second"), 2, 1, 1, 0) is True
    assert more.outcome(_m("bh_home", "yes"), 2, 1, 1, 0) is True
    assert more.outcome(_m("fh1x2", "home"), 2, 1, None, None) is None  # no half-time score
    row = pd.Series({"FTHG": 2, "FTAG": 0, "HTHG": 0, "HTAG": 0})
    assert leg_won(row, "shgoals", "over", 1.5) is True
    assert leg_won(row, "cs_home", "yes", None) is True
    assert leg_won(row, "dnb", "away", None) is False


def _leg(market: str, sel: str, line: float | None, p: float) -> Leg:
    return Leg(
        "m1",
        "E0",
        "2026-12-01T15:00:00Z",
        "A",
        "B",
        "A",
        "B",
        market,
        sel,
        line,
        f"{market} {sel}",
        p,
        "High",
        True,
    )


def test_slip_chance_reads_half_and_full_time_legs_together() -> None:
    j = more.joint(GRID, 0.44, 0.44)
    h, a, h1, a1 = np.indices(j.shape)
    want = j[(h1 > a1) & (h > a)].sum()
    legs = [_leg("fh1x2", "home", None, 0.3), _leg("1x2", "home", None, 0.5)]
    assert match_chance(legs, GRID, None, {}, (0.44, 0.44)) == pytest.approx(want)
    with pytest.raises(ValueError):
        match_chance(legs, GRID, None, {}, None)  # half legs need the shares
    dnb = _leg("dnb", "home", None, 0.66)
    assert match_chance([dnb], GRID, None, {}) == pytest.approx(0.66)
    with pytest.raises(ValueError):
        match_chance([dnb, _leg("ou25", "over", 2.5, 0.5)], GRID, None, {})


def test_goal_test_still_takes_full_time_only_for_old_markets() -> None:
    assert goal_test("1x2", "home")(np.array([2]), np.array([1]))[0]
    with pytest.raises(ValueError):
        goal_test("fh1x2", "home")(np.array([2]), np.array([1]))


def test_saved_refund_slip_settles_as_void() -> None:
    legs = [
        {
            "key": "m1|dnb|home|",
            "league": "E0",
            "kickoff_utc": "2026-12-01T15:00:00+00:00",
            "home_key": "A",
            "away_key": "B",
            "market": "dnb",
            "sel": "home",
            "line": None,
        }
    ]
    log = pd.DataFrame(
        [
            {
                "saved_on": "x",
                "slip_id": "s0001",
                "kind": "single",
                "legs": json.dumps(legs),
                "labels": "",
                "prob": 0.6,
                "risk": "Low",
                "price": 1.8,
                "stake": 10.0,
                "result": None,
                "won": None,
                "profit": None,
            }
        ]
    )
    matches = pd.DataFrame(
        {
            "league": ["E0"],
            "home": ["A"],
            "away": ["B"],
            "date": pd.to_datetime(["2026-12-01"]),
            "FTHG": [1],
            "FTAG": [1],
            "HTHG": [0],
            "HTAG": [1],
            "HC": [5],
            "AC": [4],
            "HY": [1],
            "AY": [2],
            "HR": [0],
            "AR": [0],
        }
    )
    out = settle(log, matches)
    assert out.loc[0, "won"] == "void" and out.loc[0, "profit"] == 0.0


def test_best_markets_rule_and_board_filter() -> None:
    from football_edge.trader.record import best_set

    skill = pd.DataFrame(
        {
            "market": ["1x2", "1x2", "dnb", "ou25"],
            "sel": ["home", "draw", "home", "over"],
            "line": [None, None, None, 2.5],
            "best": [True, False, True, False],
        }
    )
    assert best_set(skill) == {("1x2", "home", None), ("dnb", "home", None)}

    from football_edge.trader.desk import Desk

    d = Desk.__new__(Desk)
    d.conf = {"best_only": True}
    leg = _leg("dnb", "home", None, 0.6)
    best = Leg(**{**leg.__dict__, "best": True})
    assert not d._eligible(leg, single=True)
    assert d._eligible(best, single=True) and not d._eligible(best, single=False)
    d.conf = {"add_best": True}  # old board markets plus every best market
    old = _leg("ou25", "over", 2.5, 0.6)
    new_weak = Leg(**{**_leg("exact", "2", None, 0.3).__dict__, "board": False})
    assert d._eligible(old, single=False) and d._eligible(best, single=True)
    assert not d._eligible(new_weak, single=True)
