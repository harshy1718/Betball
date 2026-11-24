"""Live tracking: the append-only log, grading, the scorecard and the posts rule."""

import dataclasses

import pandas as pd
import pytest

from football_edge.tracking import posts, scorecard
from football_edge.tracking.ledger import Ledger
from football_edge.tracking.results import Results, leg_won
from football_edge.tracking.snapshot import PRED_COLS, ledgers, take_snapshot
from football_edge.trader.legs import Leg

KO = "2026-11-28T15:00:00+00:00"


def leg(market="1x2", sel="home", p=0.65, line=None, conf="High", mid="e0-20261128-a-b"):
    return Leg(
        mid,
        "E0",
        KO,
        "Alpha",
        "Beta",
        "A",
        "B",
        market,
        sel,
        line,
        f"{market} {sel}",
        p,
        conf,
        True,
    )


@pytest.fixture()
def tcfg(cfg, tmp_path):
    t = {
        "predictions_dir": str(tmp_path / "records/predictions"),
        "posted_dir": str(tmp_path / "records/posted"),
        "slips_dir": str(tmp_path / "bets/desk_log"),
        "min_sample": 2,
        "baseline_years": 3,
        "strong": {"min_hit": 0.65, "min_band_n": 30, "min_class_n": 100},
    }
    return dataclasses.replace(cfg, tracking=t)


def played(rows):
    """Results table: (date, home, away, home goals, away goals)."""
    df = pd.DataFrame(rows, columns=["date", "home", "away", "FTHG", "FTAG"])
    return df.assign(
        league="E0", date=pd.to_datetime(df["date"]), HC=5, AC=5, HY=2, AY=2, HR=0, AR=0,
        close_h=2.0, close_d=3.5, close_a=4.0, ou_close_over=1.9, ou_close_under=1.9,
    )  # fmt: skip


# -- the log ------------------------------------------------------------------------------
def test_ledger_appends_and_detects_any_edit(tmp_path):
    book = Ledger(tmp_path / "log", ["taken_at", "x", "p"])
    book.append(pd.DataFrame({"x": ["a", "b"], "p": ["0.5", "0.6"]}), "2026-11-22T09:00:00Z")
    book.append(pd.DataFrame({"x": ["c"], "p": ["0.7"]}), "2026-11-23T09:00:00Z")
    assert len(book.read()) == 3 and book.verify() == []
    f = tmp_path / "log" / "2026-11.csv"
    f.write_text(f.read_text().replace("0.6", "0.9"))  # quietly improve an old prediction
    assert [p.what for p in book.verify()] == ["rows were changed"]


def test_ledger_ignores_windows_line_endings(tmp_path):
    book = Ledger(tmp_path / "log", ["taken_at", "x"])
    book.append(pd.DataFrame({"x": ["a", "b"]}), "2026-11-22T09:00:00Z")
    f = tmp_path / "log" / "2026-11.csv"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))  # what git may do on Windows
    assert book.verify() == []


def test_ledger_detects_deleted_rows(tmp_path):
    book = Ledger(tmp_path / "log", ["taken_at", "x"])
    book.append(pd.DataFrame({"x": ["a", "b"]}), "2026-11-22T09:00:00Z")
    f = tmp_path / "log" / "2026-11.csv"
    f.write_text("\n".join(f.read_text().splitlines()[:2]) + "\n")  # drop a losing row
    assert "rows, chain says 2" in book.verify()[0].what


# -- grading ------------------------------------------------------------------------------
def test_leg_won_for_goals_handicap_and_counts():
    r = pd.Series({"FTHG": 0, "FTAG": 1, "HC": 4, "AC": 6, "HY": 1, "AY": 2, "HR": 0, "AR": 1})
    assert leg_won(r, "hcp", "home+1.5", None) is True  # lost by one: +1.5 still comes in
    assert leg_won(r, "1x2", "home", None) is False
    assert leg_won(r, "corners", "over", 9.5) is True  # 10 corners
    assert leg_won(r, "cards", "under", 3.5) is False  # 4 cards
    assert leg_won(r.drop("HC"), "corners", "over", 9.5) is None


def test_only_the_last_prediction_before_kickoff_counts():
    rows = pd.DataFrame({
        "taken_at": ["2026-11-27T09:00:00Z", "2026-11-28T09:00:00Z", "2026-11-28T16:00:00Z"],
        "kickoff_utc": [KO] * 3, "match_id": ["m"] * 3, "p": ["0.5", "0.6", "0.9"],
    })  # fmt: skip
    kept = scorecard.last_before_kickoff(rows, "match_id")
    assert kept["p"].tolist() == ["0.6"]  # the after-kickoff row never counts


def test_canonical_counts_each_question_once():
    assert scorecard.canonical("ou25", "over") and not scorecard.canonical("ou25", "under")
    assert not scorecard.canonical("dc", "1X")
    assert scorecard.canonical("hcp", "home-1.5") and not scorecard.canonical("hcp", "away+1.5")
    assert all(scorecard.canonical("1x2", s) for s in ("home", "draw", "away"))


def test_baseline_is_the_leagues_own_rate_before_the_match():
    m = played([("2025-01-01", "A", "B", 2, 0), ("2025-06-01", "C", "D", 0, 0),
                ("2026-11-30", "E", "F", 3, 0)])  # fmt: skip
    base = scorecard.Baseline(m, years=3)
    # before 28 Nov 2026: 1 home win in 2 matches; the later match is not seen
    assert base.rate("E0", "1x2", "home", None, pd.Timestamp(KO)) == pytest.approx(0.5)


def test_scorecard_grades_logged_predictions(tcfg):
    book = ledgers(tcfg)["predictions"]
    rows = []
    for i, (h, a) in enumerate([("A", "B"), ("C", "D"), ("E", "F")]):
        mid = f"e0-2026112{i}-{h}-{a}".lower()
        ko = f"2026-11-2{i}T15:00:00+00:00"
        for sel, p in (("home", 0.6), ("draw", 0.25), ("away", 0.15)):
            rows.append({"match_id": mid, "league": "E0", "kickoff_utc": ko, "home_key": h,
                         "away_key": a, "market": "1x2", "sel": sel, "line": "", "p": str(p),
                         "confidence": "High", "skill": "True"})  # fmt: skip
    book.append(pd.DataFrame(rows), "2026-11-19T09:00:00Z")
    results = played([("2026-11-20", "A", "B", 1, 0), ("2026-11-21", "C", "D", 0, 2),
                      ("2024-01-01", "X", "Y", 1, 1)])  # E v F not played yet  # fmt: skip
    card = scorecard.build(tcfg, results)
    assert card["logged_matches"] == 3 and card["graded_matches"] == 2
    res = card["markets"][0]
    assert res["name"] == "Match result" and res["n"] == 6  # 3 outcomes x 2 played matches
    assert res["hit"] == pytest.approx(2 / 6, abs=1e-4)
    assert res["market_n"] == 6 and res["market_brier"] is not None
    assert card["log_ok"]


def test_slips_graded_by_type(tcfg):
    preds = pd.DataFrame([{
        "taken_at": "2026-11-27T09:00:00Z", "match_id": "m1", "league": "E0", "kickoff_utc": KO,
        "home_key": "A", "away_key": "B", "market": "1x2", "sel": "home", "line": "", "p": "0.6",
        "confidence": "High", "skill": "True"}])  # fmt: skip
    slips = pd.DataFrame([
        {"taken_at": "2026-11-27T09:00:00Z", "group": "m1", "kickoff_utc": KO, "title": "Single",
         "kind": "single", "keys": "m1|1x2|home|", "labels": "", "p": "0.6", "risk": "Low"},
        {"taken_at": "2026-11-27T09:00:00Z", "group": "m1", "kickoff_utc": KO,
         "title": "Safe parlay", "kind": "parlay", "keys": "m1|1x2|home|;m1|ou25|over|2.5",
         "labels": "", "p": "0.4", "risk": "Medium"},
    ])  # fmt: skip
    res = Results(played([("2026-11-28", "A", "B", 1, 0)]))  # home win, 1 goal: under 2.5
    out = {s["title"]: s for s in scorecard.slip_record(slips, preds, res)}
    assert out["Single"]["won"] == 1 and out["Safe parlay"]["won"] == 0
    assert out["Safe parlay"]["floor"] == 0.60


# -- snapshots ----------------------------------------------------------------------------
class FakeDesk:
    def __init__(self, p=0.65):
        self.p = p

    def legs_by_match(self):
        m = {"id": "e0-20261128-a-b", "kickoff_utc": KO}
        return [m], {m["id"]: [leg(p=self.p), leg("ou25", "over", 0.55, 2.5)]}

    def board(self, now=None):
        legd = {"key": "e0-20261128-a-b|1x2|home|", "kickoff_utc": KO, "label": "Alpha win",
                "home": "Alpha", "away": "Beta"}  # fmt: skip
        s = {"title": "Single", "kind": "single", "legs": [legd], "p": self.p, "risk": "Low"}
        return {"top": [], "matches": [{"match": {"id": "e0-20261128-a-b"}, "singles": [s],
                                        "parlays": {"safe": None, "risky": None}}]}  # fmt: skip


def test_snapshot_writes_only_when_the_numbers_move(tcfg):
    now = pd.Timestamp("2026-11-27T09:00:00Z")
    assert take_snapshot(tcfg, FakeDesk(), now) == {"predictions": 2, "slips": 1}
    assert take_snapshot(tcfg, FakeDesk(), now + pd.Timedelta(hours=1)) == {
        "predictions": 0,
        "slips": 0,
    }
    assert take_snapshot(tcfg, FakeDesk(0.70), now + pd.Timedelta(hours=2))["predictions"] == 2
    df = ledgers(tcfg)["predictions"].read()
    assert list(df.columns) == PRED_COLS and ledgers(tcfg)["predictions"].verify() == []


def test_snapshot_skips_matches_that_kicked_off(tcfg):
    late = pd.Timestamp(KO) + pd.Timedelta(minutes=5)
    assert take_snapshot(tcfg, FakeDesk(), late)["predictions"] == 0


# -- posts --------------------------------------------------------------------------------
RECORD = pd.DataFrame({
    "source": ["model"] * 6, "market": ["1x2"] * 4 + ["ou25"] * 2,
    "band": [0.55, 0.60, 0.65, 0.70, 0.70, 0.75],
    "n": [700, 600, 400, 300, 200, 40],
    "hit_rate": [0.57, 0.69, 0.71, 0.74, 0.71, 0.60],
})  # fmt: skip


def test_strong_level_comes_from_the_track_record():
    lv = posts.strong_levels(RECORD, {"min_hit": 0.65, "min_band_n": 30, "min_class_n": 100})
    assert lv["1x2"]["min_p"] == 0.60  # 55% level came true only 57%
    assert lv["1x2"]["n"] == 1300
    assert "ou25" not in lv  # the 75% goals level (40 calls) came true only 60%


def test_strong_calls_never_include_a_draw():
    lv = posts.strong_levels(RECORD, {})
    assert posts.strong_calls([leg(sel="draw", p=0.7)], lv) == []
    calls = posts.strong_calls([leg(p=0.62), leg(sel="away", p=0.1)], lv)
    assert [c["call"] for c in calls] == ["Alpha to win"]
    assert posts.strong_calls([leg(p=0.58)], lv) == []


def test_caption_has_no_betting_language():
    probs = {"home": 0.62, "draw": 0.2, "away": 0.18}
    m = {"home": "Alpha", "away": "Beta", "league": "E0", "league_name": "Premier League",
         "day_ist": "Sat 28 Nov", "time_ist": "20:30", "probs": probs,
         "top_score": {"score": "2-0"}, "exp_goals": {"home": 1.9, "away": 0.7}}  # fmt: skip
    text = posts.caption(m, posts.strong_calls([leg(p=0.62)], posts.strong_levels(RECORD, {})))
    assert "Alpha to win (62%)" in text and posts.betting_words(text) == []
    assert posts.betting_words("best odds tonight") == ["odds"]


class PostsDesk(FakeDesk):
    class svc:  # noqa: N801 - mimics the desk's NoxService attribute
        record = RECORD


def test_mark_posted_once_and_only_before_kickoff(tcfg):
    desk, before = PostsDesk(), pd.Timestamp("2026-11-27T09:00:00Z")
    assert posts.mark_posted(tcfg, desk, "e0-20261128-a-b", "result", before) == {"already": False}
    assert posts.mark_posted(tcfg, desk, "e0-20261128-a-b", "result", before) == {"already": True}
    with pytest.raises(posts.NotPostableError, match="not a strong call"):
        posts.mark_posted(tcfg, desk, "e0-20261128-a-b", "goals", before)
    with pytest.raises(posts.NotPostableError, match="kicked off"):
        posts.mark_posted(tcfg, desk, "e0-20261128-a-b", "result", pd.Timestamp(KO))
    assert len(ledgers(tcfg)["posted"].read()) == 1


def test_recap_counts_hits_and_misses(tcfg):
    posts.mark_posted(
        tcfg, PostsDesk(), "e0-20261128-a-b", "result", pd.Timestamp("2026-11-27T09:00Z")
    )
    lost = played([("2026-11-28", "A", "B", 0, 1)])
    r = posts.recap(tcfg, lost, now=pd.Timestamp("2026-11-30", tz="UTC"))
    assert (r["settled"], r["won"], r["calls"][0]["won"], r["calls"][0]["score"]) == (
        1,
        0,
        False,
        "0-1",
    )


def test_tracking_routes_are_local_only(tcfg, tmp_path):
    from fastapi.testclient import TestClient

    from football_edge.web.app import create_app

    app = create_app(tcfg, dist=tmp_path, desk=PostsDesk())
    outside = TestClient(app, client=("203.0.113.9", 50000))
    for path in ("/api/desk/scorecard", "/api/posts", "/api/posts/recap"):
        assert outside.get(path).status_code == 403
    assert (
        outside.post("/api/posts/mark", json={"match_id": "x", "kind": "result"}).status_code == 403
    )
