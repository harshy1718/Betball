import pandas as pd

from football_edge.config import OddsSource
from football_edge.data.loader import infer_season, pick_odds


def test_infer_season_autumn_start():
    assert infer_season(pd.Series(pd.to_datetime(["2019-08-09", "2020-07-26"]))) == 2019


def test_pick_odds_falls_back_per_row():
    df = pd.DataFrame(
        {
            "PSCH": [2.0, None],
            "PSCD": [3.0, None],
            "PSCA": [4.0, None],
            "AvgCH": [2.1, 2.2],
            "AvgCD": [3.1, 3.2],
            "AvgCA": [4.1, 4.2],
        }
    )
    pri = (
        OddsSource("PS", ("PSCH", "PSCD", "PSCA")),
        OddsSource("Avg", ("AvgCH", "AvgCD", "AvgCA")),
    )
    out = pick_odds(df, pri)
    assert out["src"].tolist() == ["PS", "Avg"]
    assert out.loc[1, "h"] == 2.2


def test_file_names_of_every_league_are_recognised():
    from football_edge.data.loader import _FILE_RE

    for f in ("E0_2015.csv", "I1_2020.csv", "SP1_2026.csv", "D1_1516.csv", "F1_2627.csv"):
        assert _FILE_RE.match(f), f
    for f in ("E1_2015.csv", "SC0_2020.csv", "notes.csv"):
        assert not _FILE_RE.match(f), f


def test_golden_bars_stay_in_their_leagues():
    import pandas as pd

    from football_edge.betting import golden

    fc = pd.DataFrame({"league": ["E0", "D1"], "date": ["2026-12-01"] * 2, "home": ["A", "C"],
                       "away": ["B", "D"], "p_h": [0.62, 0.62], "p_d": [0.2, 0.2],
                       "p_a": [0.18, 0.18], "top_score": ["1-0", "1-0"],
                       "top_score_p": [0.13, 0.13]})  # fmt: skip
    gb = {"leagues": ["E0"], "lo": 0.60, "hi": 0.64}
    assert list(golden.golden_picks(fc, gb)["league"]) == ["E0"]
    assert list(golden.cs_picks(fc, 0.12, ["E0"])["league"]) == ["E0"]


def test_golden_bars_use_the_unadjusted_chances_they_were_searched_on():
    import pandas as pd

    from football_edge.betting import golden

    fc = pd.DataFrame({"league": ["E0"], "date": ["2026-12-01"], "home": ["A"], "away": ["B"],
                       "p_h": [0.66], "p_d": [0.2], "p_a": [0.14],
                       "raw_p_h": [0.61], "raw_p_d": [0.22], "raw_p_a": [0.17],
                       "top_score": ["1-1"], "top_score_p": [0.125],
                       "raw_top_score": ["1-0"], "raw_top_score_p": [0.115]})  # fmt: skip
    g = golden.golden_picks(fc, {"leagues": ["E0"], "lo": 0.60, "hi": 0.64})
    assert len(g) == 1 and g.iloc[0]["prob"] == 0.61  # 66% calibrated would be outside
    assert golden.cs_picks(fc, 0.12).empty  # raw 11.5% misses the bar; calibrated 12.5% would not


def test_correct_score_bar_has_a_minimum_price_per_league():
    import pandas as pd

    from football_edge.betting import golden

    cs = {"lo": 0.12, "leagues": ["E0", "D1"], "min_odds": {"E0": 8.0, "D1": 8.5}}
    assert golden.cs_min_odds(cs, "E0") == 8.0
    assert golden.cs_min_odds(cs, "I1") == golden.CS_MIN_ODDS  # no setting: the old fallback
    f = pd.Series({"league": "D1", "top_score": "1-1", "top_score_p": 0.13})
    assert "take at 8.50+" in golden.flags(f, {"golden_cs": cs})["cs"]
