import pandas as pd

from football_edge.betting import golden


def _fc():
    return pd.DataFrame({
        "league": "E0", "date": pd.to_datetime(["2026-11-21"] * 3),
        "home": ["A", "C", "E"], "away": ["B", "D", "F"],
        "p_h": [0.62, 0.70, 0.20], "p_d": [0.22, 0.18, 0.18], "p_a": [0.16, 0.12, 0.62],
        "odds_h": [1.55, 1.40, 5.0], "odds_d": [4.0, 4.5, 4.2], "odds_a": [6.0, 8.0, 1.60],
    })  # fmt: skip


def test_picks_log_settle_and_score(tmp_path):
    gp = golden.golden_picks(_fc(), 0.60, 0.64)
    assert list(gp["label"]) == ["A win", "F win"]  # 70% is outside the bar
    path = tmp_path / "golden.csv"
    today = pd.Timestamp("2026-11-21")
    assert golden.log_picks(gp, path, today) == 2
    assert golden.log_picks(gp, path, today) == 0  # no duplicates
    results = pd.DataFrame({"league": "E0", "date": pd.to_datetime(["2026-11-21"] * 2),
                            "home": ["A", "E"], "away": ["B", "F"],
                            "FTHG": [2, 1], "FTAG": [0, 1]})  # fmt: skip
    log = golden.settle(path, results)
    assert list(log["result"]) == ["2-0", "1-1"]
    assert list(log["profit"].round(2)) == [55.0, -100.0]
    assert "2 of 200" in golden.scoreboard(log)


def test_correct_score_bar(tmp_path):
    fc = _fc()
    fc["top_score"] = ["1-0", "2-0", "0-1"]
    fc["top_score_p"] = [0.13, 0.11, 0.125]
    cp = golden.cs_picks(fc, 0.12)
    assert list(cp["pick"]) == ["1-0", "0-1"]
    path = tmp_path / "cs.csv"
    golden.log_picks(cp, path, pd.Timestamp("2026-11-21"))
    results = pd.DataFrame({"league": "E0", "date": pd.to_datetime(["2026-11-21"] * 2),
                            "home": ["A", "E"], "away": ["B", "F"],
                            "FTHG": [1, 1], "FTAG": [0, 1]})  # fmt: skip
    log = golden.settle(path, results)
    assert list(log["won"]) == [True, False] and log["profit"].isna().all()  # no price typed
    assert "break-even price 2.00" in golden.scoreboard(log, "Correct-score bar")


def test_set_price(tmp_path):
    fc = _fc()
    fc["top_score"], fc["top_score_p"] = ["1-0", "2-0", "0-1"], [0.13, 0.11, 0.125]
    path = tmp_path / "cs.csv"
    golden.log_picks(golden.cs_picks(fc, 0.12), path, pd.Timestamp("2026-11-21"))
    assert golden.set_price(path, "f", 8.0) == ["E v F: 0-1 at 8.00"]
    assert golden.read(path)["site_odds"].tolist()[1] == 8.0
    assert golden.set_price(path, "zzz", 8.0) == []
