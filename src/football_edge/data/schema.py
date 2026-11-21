"""Contract for the processed matches table, and validation of it."""

from __future__ import annotations

import pandas as pd

REQUIRED = ("league", "season", "date", "home", "away", "FTHG", "FTAG", "FTR")
ODDS = (
    "close_h",
    "close_d",
    "close_a",
    "pre_h",
    "pre_d",
    "pre_a",
    "max_h",
    "max_d",
    "max_a",
    "maxc_h",
    "maxc_d",
    "maxc_a",
    "ou_close_over",
    "ou_close_under",
)


class DataValidationError(ValueError):
    """Raised when data violates the processed-table contract."""


def validate_matches(df: pd.DataFrame) -> None:
    problems: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    for c in ("date", "home", "away", "FTHG", "FTAG", "FTR"):
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{n} null values in {c}")
    if (df["home"] == df["away"]).any():
        problems.append("rows where home == away")
    if df.duplicated(["league", "date", "home", "away"]).any():
        problems.append("duplicate fixtures")
    if (df[["FTHG", "FTAG"]] < 0).any().any():
        problems.append("negative goals")
    expected = pd.Series("D", index=df.index)
    expected[df["FTHG"] > df["FTAG"]] = "H"
    expected[df["FTHG"] < df["FTAG"]] = "A"
    if (expected != df["FTR"]).any():
        problems.append("FTR inconsistent with goals")
    for c in ODDS:
        if c in df.columns and (df[c].dropna() <= 1.0).any():
            problems.append(f"decimal odds <= 1.0 in {c}")
    if problems:
        raise DataValidationError("; ".join(problems))
