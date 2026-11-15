import pandas as pd
import pytest

from football_edge.data.schema import DataValidationError, validate_matches


def test_valid_frame_passes(tiny_matches):
    validate_matches(tiny_matches)


def test_inconsistent_result_rejected(tiny_matches):
    tiny_matches.loc[0, "FTR"] = "A"
    with pytest.raises(DataValidationError, match="FTR"):
        validate_matches(tiny_matches)


def test_duplicate_fixture_rejected(tiny_matches):
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_matches(pd.concat([tiny_matches, tiny_matches.iloc[[0]]]))


def test_bad_odds_rejected(tiny_matches):
    tiny_matches.loc[0, "close_h"] = 0.9
    with pytest.raises(DataValidationError, match="odds"):
        validate_matches(tiny_matches)


def test_missing_column_rejected(tiny_matches):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_matches(tiny_matches.drop(columns=["home"]))
