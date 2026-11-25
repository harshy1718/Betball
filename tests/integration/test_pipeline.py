import pytest

from football_edge.data.loader import load_matches

pytestmark = pytest.mark.integration


def test_processed_table_is_valid_and_sized(cfg):
    df = load_matches(cfg)
    assert set(df.league) == {"E0", "I1", "SP1", "D1", "F1"}
    sizes = df[df.season.between(2015, 2025)].groupby("league").size().to_dict()
    # 20 teams in England, Italy, Spain; 18 in Germany; France 20 until 2022/23 (2019/20 stopped
    # in March: 279 matches), then 18
    assert sizes == {"E0": 4180, "I1": 4180, "SP1": 4180, "D1": 11 * 306, "F1": 3857}
