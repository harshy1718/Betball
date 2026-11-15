import numpy as np
import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.runner import walk_forward


class Uniform:
    name = "uniform"

    def __init__(self):
        self.seen_results_at_predict = None

    def fit(self, train):
        pass

    def predict_proba(self, test):
        Uniform.last_test_columns = set(test.columns)
        return np.full((len(test), 3), 1 / 3)


def _frame():
    return pd.DataFrame(
        {
            "league": ["E0"] * 4,
            "season": [2018] * 4,
            "date": pd.to_datetime(["2018-08-01", "2018-08-02", "2018-08-08", "2018-08-09"]),
            "home": ["A", "B", "A", "C"],
            "away": ["B", "C", "C", "A"],
            "FTHG": [1, 0, 2, 1],
            "FTAG": [0, 0, 1, 3],
            "FTR": ["H", "D", "H", "A"],
        }
    )


def test_walk_forward_one_row_per_test_match_and_hides_results():
    m = _frame()
    folds = [Fold(pd.Timestamp("2018-08-06"), np.array([0, 1]), np.array([2, 3]))]
    preds = walk_forward(m, folds, Uniform)
    assert preds["row"].tolist() == [2, 3]
    assert preds["outcome"].tolist() == [0, 2]
    assert not {"FTR", "FTHG", "FTAG"} & Uniform.last_test_columns
