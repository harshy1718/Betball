import json

import numpy as np
import pandas as pd
import pytest

from football_edge.trader.brain import facts, review
from football_edge.trader.desk import desk_view, settle
from football_edge.trader.legs import Leg, find_leg
from football_edge.trader.slip import Book, History, check, match_chance

STAKING = {
    "monthly_budget": 10000,
    "currency": "INR",
    "stake_rounding": 10,
    "stop_loss_fraction": 0.30,
}
TRADER = {"stake_by_risk": {"Low": 0.02, "Medium": 0.01, "High": 0.005, "Very high": 0.0025}}


def leg(mid="m1", market="1x2", sel="home", p=0.6, line=None, conf="High", skill=True, label=None):
    return Leg(
        match_id=mid,
        league="E0",
        kickoff_utc="2026-11-22T16:30:00+00:00",
        home="Liverpool",
        away="Manchester United",
        home_key="Liverpool",
        away_key="Man United",
        market=market,
        sel=sel,
        line=line,
        label=label or f"{market} {sel} {line or ''}".strip(),
        p=p,
        confidence=conf,
        skill=skill,
    )


# a tiny score grid: P(1-0)=0.3, P(1-1)=0.3, P(2-1)=0.2, P(0-2)=0.2
GRID = np.zeros((3, 3))
GRID[1, 0], GRID[1, 1], GRID[2, 1], GRID[0, 2] = 0.3, 0.3, 0.2, 0.2


def test_goal_legs_in_one_match_use_the_score_grid():
    home = leg(p=0.5)  # 1-0 and 2-1
    over = leg(market="ou25", sel="over", p=0.2, line=2.5)  # only 2-1 has 3+ goals
    assert match_chance([home], GRID, None, {}) == pytest.approx(0.5)
    assert match_chance([home, over], GRID, None, {}) == pytest.approx(0.2)  # not 0.5 x 0.2
    away = leg(sel="away", p=0.2)
    assert match_chance([home, away], GRID, None, {}) == 0.0


def test_count_legs_become_a_range():
    cp = {("corners", 8.5): 0.6, ("corners", 10.5): 0.4, ("corners", 11.5): 0.3}
    over, under = (
        leg(market="corners", sel="over", line=8.5, p=0.6),
        leg(market="corners", sel="under", line=11.5, p=0.7),
    )
    assert match_chance([over, under], None, None, cp) == pytest.approx(0.3)  # 9 to 11
    clash = leg(market="corners", sel="under", line=8.5, p=0.4)
    assert match_chance([over, clash], None, None, cp) == 0.0


def test_lift_from_history_adjusts_goal_and_corner_legs():
    # history: corners 10+ happened in all 3 home wins and none of the 3 others
    hist = History(
        hg=np.array([1, 2, 1, 0, 0, 1]),
        ag=np.array([0, 0, 0, 1, 0, 1]),
        totals={"corners": np.array([12, 11, 10, 5, 6, 7.0]), "cards": np.zeros(6)},
    )
    corners = leg(market="corners", sel="over", line=9.5, p=0.5)
    import football_edge.trader.slip as slip

    old, slip.MIN_LIFT_N = slip.MIN_LIFT_N, 1
    try:
        p = match_chance([leg(p=0.5), corners], GRID, hist, {("corners", 9.5): 0.5})
    finally:
        slip.MIN_LIFT_N = old
    assert p == pytest.approx(0.375)  # lift 2 is capped at 1.5: 0.5 x 0.5 x 1.5


def test_check_grades_risk_and_costs():
    book = Book({}, {}, None)
    a, b = leg(mid="m1", p=0.8, label="A"), leg(mid="m2", p=0.7, label="B")
    c = check([a, b], book, STAKING, TRADER)
    assert c["p"] == pytest.approx(0.56) and c["risk"] == "Medium" and c["kind"] == "parlay"
    assert c["stake"] == 100 and c["fair_odds"] == pytest.approx(1 / 0.56)
    assert {x["label"]: x["without"] for x in c["leg_costs"]} == pytest.approx({"A": 0.7, "B": 0.8})
    weak = leg(mid="m3", market="corners", sel="over", line=9.5, p=0.9, skill=False, label="C")
    c2 = check([a, weak], book, STAKING, TRADER)
    assert c2["risk"] == "Medium" and c2["weak_legs"] == ["C"]  # 72% is Low, +1 for the weak leg
    with pytest.raises(ValueError):
        check([a, a], book, STAKING, TRADER)


def test_stake_stops_at_the_stop_loss_and_price_verdict():
    book = Book({}, {}, None)
    c = check([leg(p=0.65)], book, STAKING, TRADER, month_profit=-3500)
    assert c["stake"] == 0 and "stop-loss" in c["stake_note"]
    c = check([leg(p=0.5)], book, STAKING, TRADER, price=2.2)
    assert c["price"]["verdict"] == "above our fair price"
    assert c["price"]["back_per_100"] == pytest.approx(110)
    c = check([leg(p=0.5)], book, STAKING, TRADER, price=1.8)
    assert desk_view({**c, "desk_view": ""}).startswith("Skip at this price")


def test_find_leg_reads_bets_in_words():
    legs = [
        leg(label="Liverpool win"),
        leg(market="dc", sel="1X"),
        leg(market="ou25", sel="over", line=2.5),
        leg(market="btts", sel="no"),
        leg(market="corners", sel="over", line=9.5),
        leg(market="cards", sel="under", line=4.5),
        leg(market="score", sel="2-1"),
    ]
    assert find_leg(legs, "Liverpool win").market == "1x2"
    assert find_leg(legs, "1x").sel == "1X"
    assert find_leg(legs, "over 2.5 goals").market == "ou25"
    assert find_leg(legs, "BTTS no").sel == "no"
    assert find_leg(legs, "over 9.5 corners").line == 9.5
    assert find_leg(legs, "cards under 4.5").market == "cards"
    assert find_leg(legs, "2-1").sel == "2-1"
    assert find_leg(legs, "first goalscorer") is None


def test_saved_slips_settle_from_results():
    legs = [
        leg(label="Liverpool win").to_dict() | {"key": "k1"},
        leg(market="corners", sel="over", line=9.5).to_dict() | {"key": "k2"},
    ]
    log = pd.DataFrame(
        [
            {
                "saved_on": "2026-11-22",
                "slip_id": "s0001",
                "kind": "same-match",
                "legs": json.dumps(legs),
                "labels": "",
                "prob": 0.3,
                "risk": "High",
                "price": 4.0,
                "stake": 50,
                "result": None,
                "won": None,
                "profit": None,
            }
        ]
    )
    m = pd.DataFrame(
        {
            "league": ["E0"],
            "home": ["Liverpool"],
            "away": ["Man United"],
            "date": pd.to_datetime(["2026-11-22"]),
            "FTHG": [2],
            "FTAG": [1],
            "HC": [6],
            "AC": [5],
            "HY": [1],
            "AY": [2],
            "HR": [0],
            "AR": [0],
        }
    )
    out = settle(log.copy(), m)
    assert out.at[0, "won"] is True and out.at[0, "profit"] == pytest.approx(150)
    assert out.at[0, "result"] == "2-1, 2-1"
    m.loc[0, "HC"] = 3  # 8 corners: the corners leg loses
    out = settle(log.copy(), m)
    assert out.at[0, "won"] is False and out.at[0, "profit"] == -50


class Scripted:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, tools):
        self.sent.append(list(messages))
        return {"role": "assistant", "content": self.replies.pop(0)}, {}


def _checked():
    c = check(
        [leg(mid="m1", p=0.8, label="A"), leg(mid="m2", p=0.7, label="B")],
        Book({}, {}, None),
        STAKING,
        TRADER,
    )
    c["desk_view"] = desk_view(c)
    return c


def test_review_quotes_only_the_facts():
    c = _checked()
    assert "Chance the whole slip wins: 56%" in facts(c)
    r = review(Scripted(["Medium risk at 56%. Without B it is 80%. Stake 100 INR."]), c)
    assert r["checked"] and r["text"].startswith("Medium")
    r = review(Scripted(["Take it, 90% sure.", "Medium risk at 56%."]), c)
    assert r["text"] == "Medium risk at 56%." and r["checked"]


def test_review_never_shows_a_promise():
    c = _checked()
    r = review(Scripted(["A guaranteed winner.", "Still a lock."]), c)
    assert r["text"].startswith(c["desk_view"]) and "lock" not in r["text"].lower()


def test_desk_only_answers_this_pc(cfg, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from football_edge.web.app import create_app

    class StubDesk:
        def board(self):
            return {"top": [{"title": "Safest single"}], "golden": [], "matches": []}

    app = create_app(cfg, dist=tmp_path, desk=StubDesk())
    assert TestClient(app).get("/api/desk/board").json()["top"] == [{"title": "Safest single"}]
    outside = TestClient(app, client=("203.0.113.9", 50000))
    assert outside.get("/api/desk/board").status_code == 403


def test_corners_never_count_as_skilled():
    from football_edge.trader.legs import LegFactory

    rec = pd.DataFrame(
        {
            "source": ["model", "model"],
            "market": ["corners_9.5", "cards_4.5"],
            "band": [0.5, 0.5],
            "n": [900, 900],
            "predicted": [0.52, 0.52],
            "hit_rate": [0.52, 0.52],
            "skill": [True, True],
            "label": ["High", "High"],
        }
    )
    f = LegFactory(rec)
    assert f.trust("corners_9.5", 0.52) == ("High", False)
    assert f.trust("cards_4.5", 0.52) == ("High", True)


def test_correct_score_is_a_single_only():
    book = Book({}, {}, None)
    score = leg(market="score", sel="1-1", p=0.12, conf="Low", skill=False)
    assert check([score], book, STAKING, TRADER)["kind"] == "single"
    with pytest.raises(ValueError, match="never part of a parlay"):
        check([score, leg(mid="m2")], book, STAKING, TRADER)


def test_extra_goal_bets_read_from_the_grid():
    from football_edge.trader.legs import goal_test, record_key

    assert record_key("hcp", "home-1.5", None) == "hcp_home2"
    assert record_key("hcp", "away+1.5", None) == "hcp_home2"  # the other side of home -1.5
    assert record_key("hcp", "home+2.5", None) == "hcp_away3"
    assert record_key("home_goals", "over", 1.5) == "home_goals_1.5"
    h, a = np.array([2, 1, 0, 3]), np.array([0, 1, 0, 1])
    assert goal_test("hcp", "home-1.5")(h, a).tolist() == [True, False, False, True]
    assert goal_test("hcp", "away+1.5")(h, a).tolist() == [False, True, True, False]
    assert goal_test("wtn", "home")(h, a).tolist() == [True, False, False, False]
    assert goal_test("goals", "over", 1.5)(h, a).tolist() == [True, True, False, True]
    assert goal_test("away_goals", "under", 0.5)(h, a).tolist() == [True, False, True, False]
    assert goal_test("oddeven", "odd")(h, a).tolist() == [False, False, False, False]
    # in one match: home -1.5 and home win to nil together = 2-0, 3-0... read from the grid
    hcp = leg(market="hcp", sel="home-1.5", p=0.2)
    wtn = leg(market="wtn", sel="home", p=0.3)
    grid = np.zeros((4, 4))
    grid[2, 0], grid[1, 0], grid[3, 1], grid[1, 1] = 0.2, 0.3, 0.1, 0.4
    assert match_chance([hcp, wtn], grid, None, {}) == pytest.approx(0.2)


def test_goal_record_has_every_extra_market():
    from football_edge.trader.record import build_goal_record

    rng = np.random.default_rng(0)
    n = 400
    preds = pd.DataFrame(
        {
            "season": 2021,
            "exp_home": rng.uniform(0.8, 2.2, n),
            "exp_away": rng.uniform(0.6, 1.8, n),
            "goals_home": rng.poisson(1.5, n),
            "goals_away": rng.poisson(1.1, n),
        }
    )
    rec = build_goal_record(preds, rho=-0.03)
    assert {"goals_1.5", "home_goals_0.5", "hcp_home2", "hcp_away3", "wtn", "oddeven"} <= set(
        rec["market"]
    )
    assert set(rec["label"]) <= {"High", "Medium", "Low"}
    assert (rec.groupby("market")["n"].sum() == 2 * n).all()


def test_factory_uses_the_sites_lines():
    from football_edge.trader.legs import LegFactory

    f = LegFactory(None, count_lines={"cards": (3.5, 4.5, 5.5), "corners": (8.5, 9.5, 10.5)})
    assert f.count_lines["cards"] == (3.5, 4.5, 5.5)


def _desk(conf=None):
    from football_edge.trader.desk import Desk

    d = Desk.__new__(Desk)
    d.conf = {"parlay_min_cost": 0.05, "parlay_pool": 12, **(conf or {})}
    book = Book({}, {}, None)
    d._state = lambda: {"book": book}  # type: ignore[method-assign]
    return d


def test_best_parlay_is_the_biggest_payer_above_the_floor():
    a, b, c = (
        leg(mid="m1", p=0.8, label="A"),
        leg(mid="m2", p=0.75, label="B"),
        leg(mid="m3", p=0.7, label="C"),
    )
    best = _desk().best_parlay([a, b, c])
    assert [x.label for x in best] == ["A", "B", "C"]  # 42%: still above 40%, pays the most
    d = leg(mid="m4", p=0.6, label="D")
    best = _desk().best_parlay([a, b, c, d])
    assert {x.label for x in best} == {"A", "B", "C"}  # any mix with D falls under 40%
    assert _desk().best_parlay([a, b, c], floor=0.9) is None
    assert len(_desk().best_parlay([a, b, c], floor=0.55)) == 2  # 60%: the 2-leg payer


def test_risky_uses_high_confidence_legs_only():
    a, b = leg(mid="m1", p=0.8, label="A"), leg(mid="m2", p=0.75, label="B")
    med = leg(mid="m3", p=0.7, label="M", conf="Medium")
    assert "M" in {x.label for x in _desk().best_parlay([a, b, med], floor=0.25)}
    risky = _desk().best_parlay([a, b, med], floor=0.25, max_legs=5, high_only=True)
    assert {x.label for x in risky} == {"A", "B"}


def test_board_kinds_default_and_override():
    d = _desk({"parlays": {"safe": {"floor": 0.7}}})
    k = d.kinds()
    assert k["safe"]["floor"] == 0.7 and k["safe"]["max_legs"] == 4
    assert k["risky"] == {"floor": 0.25, "max_legs": 5, "high_only": True}


def test_no_passengers_and_one_leg_per_family():
    from football_edge.trader.desk import family

    a, b = leg(mid="m1", p=0.8, label="A"), leg(mid="m2", p=0.75, label="B")
    passenger = leg(mid="m3", p=0.97, label="P")  # removing it moves the chance < 5 points
    assert "P" not in {x.label for x in _desk().best_parlay([a, b, passenger])}
    assert family("dc") == family("hcp") == "result" and family("goals") == "total"
    same = leg(mid="m1", market="dc", sel="1X", p=0.9, label="DC")  # same match, same family as A
    best = _desk().best_parlay([a, same])
    assert best is None
