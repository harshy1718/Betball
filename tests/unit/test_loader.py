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
