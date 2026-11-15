import pytest

from football_edge.data.loader import load_matches

pytestmark = pytest.mark.integration


def test_processed_table_is_valid_and_sized(cfg):
    df = load_matches(cfg)
    assert set(df.league) == {"E0", "I1", "SP1"}
    assert len(df[df.season.between(2015, 2025)]) == 3 * 11 * 380
