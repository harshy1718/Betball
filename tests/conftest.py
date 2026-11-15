from pathlib import Path

import pandas as pd
import pytest

from football_edge.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture()
def tiny_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": ["E0", "E0"],
            "season": [2020, 2020],
            "date": pd.to_datetime(["2020-09-12", "2020-09-13"]),
            "home": ["A", "C"],
            "away": ["B", "D"],
            "FTHG": [2, 0],
            "FTAG": [1, 0],
            "FTR": ["H", "D"],
            "close_h": [1.8, 2.5],
            "close_d": [3.6, 3.2],
            "close_a": [4.5, 2.9],
        }
    )
