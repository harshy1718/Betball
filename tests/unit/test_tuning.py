import numpy as np
import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.tuning import grid_search


def test_grid_search_only_uses_tuning_seasons():
    m = pd.DataFrame(
        {
            "league": ["E0"] * 6,
            "season": [2017, 2017, 2018, 2018, 2020, 2020],
            "date": pd.to_datetime(
                ["2017-08-01", "2017-08-02", "2018-08-08", "2018-08-09", "2020-09-01", "2020-09-02"]
            ),
            "home": list("ABABAB"),
            "away": list("BABABA"),
            "FTHG": [1, 0, 2, 1, 0, 3],
            "FTAG": [0, 0, 1, 1, 2, 0],
            "FTR": ["H", "D", "H", "D", "A", "H"],
        }
    )
    folds = [
        Fold(pd.Timestamp("2018-08-06"), np.array([0, 1]), np.array([2, 3])),
        Fold(pd.Timestamp("2020-08-31"), np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]
    table = grid_search(m, folds, "team_blind", {}, {}, tune_seasons=(2018, 2019))
    assert len(table) == 1
    assert table.loc[0, "n"] == 2  # only the 2018 fold was scored
