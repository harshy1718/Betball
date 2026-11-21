import pandas as pd
import pytest

from football_edge.predict.fixtures import FixtureError, load_fixtures, season_of

KNOWN = {"E0": {"Man United", "Arsenal", "Leeds"}, "SP1": {"Ath Madrid", "Getafe"}}
LEAGUES = ("E0", "I1", "SP1")


@pytest.fixture()
def aliases(tmp_path):
    p = tmp_path / "aliases.yaml"
    p.write_text("E0:\n  Manchester United: Man United\nSP1:\n  Atletico Madrid: Ath Madrid\n")
    return p


def _write(tmp_path, text):
    p = tmp_path / "fx.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_site_format_with_aliases_and_league_names(tmp_path, aliases):
    p = _write(
        tmp_path,
        "league,date,home,away,odds_h,odds_d,odds_a\n"
        "EPL,22/11/2026,arsenal,Manchester United,1.55,4.2,5.5\n"
        "La Liga,03/01/2027,Atletico Madrid,Getafe,0,,\n",
    )
    fx = load_fixtures(p, KNOWN, LEAGUES, aliases_path=aliases)
    assert fx[["home", "away"]].values.tolist() == [
        ["Arsenal", "Man United"],
        ["Ath Madrid", "Getafe"],
    ]
    assert fx["league"].tolist() == ["E0", "SP1"]
    assert fx["season"].tolist() == [2026, 2026]  # January belongs to the season that began in 2026
    assert fx["odds_h"].iloc[0] == 1.55
    assert pd.isna(fx["odds_h"].iloc[1])  # 0 is a placeholder, not a price
    assert fx["odds_btts_yes"].isna().all()  # missing columns become "no price"


def test_unknown_team_stops_with_suggestion(tmp_path):
    p = _write(tmp_path, "league,date,home,away\nE0,22/11/2026,Arsenl,Leeds\n")
    with pytest.raises(FixtureError, match="Arsenal"):
        load_fixtures(p, KNOWN, LEAGUES)
    fx = load_fixtures(p, KNOWN, LEAGUES, allow_unknown=True)
    assert fx["home"].iloc[0] == "Arsenl"


def test_football_data_format(tmp_path):
    p = _write(
        tmp_path,
        "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,PSH,PSD,PSA\n"
        "E0,22/11/2026,15:00,Arsenal,Leeds,1.4,4.8,7.5,1.6,2.3,1.42,4.9,7.9\n"
        "D1,22/11/2026,15:00,Bayern,Koln,1.2,6,11,1.4,2.9,1.2,6,11\n",
    )
    fx = load_fixtures(p, KNOWN, LEAGUES)
    assert len(fx) == 1  # other leagues dropped
    assert fx[["odds_h", "odds_over25"]].iloc[0].tolist() == [1.4, 1.6]
    assert fx["ref_h"].iloc[0] == 1.42  # market view: Pinnacle first
    assert fx["ref_over25"].iloc[0] == 1.6  # no Pinnacle over/under -> market average
    ps = load_fixtures(p, KNOWN, LEAGUES, odds_source="PS")
    assert ps["odds_h"].iloc[0] == 1.42
    assert pd.isna(ps["odds_over25"].iloc[0])  # no P>2.5 column in this file


def test_no_fixtures_for_our_leagues(tmp_path):
    p = _write(tmp_path, "league,date,home,away\nD1,22/11/2026,Bayern,Koln\n")
    with pytest.raises(FixtureError, match="no fixtures"):
        load_fixtures(p, KNOWN, LEAGUES)


def test_season_of():
    d = pd.Series(pd.to_datetime(["2026-07-01", "2026-06-30", "2027-01-15"]))
    assert season_of(d).tolist() == [2026, 2025, 2026]


def test_single_fixture_finds_league_and_resolves_aliases(aliases):
    from football_edge.predict.fixtures import single_fixture

    fx = single_fixture(
        "arsenal",
        "Manchester United",
        KNOWN,
        LEAGUES,
        date="2026-11-22",
        odds_1x2=(1.55, 4.2, 5.5),
        aliases_path=aliases,
    )
    row = fx.iloc[0]
    assert (row.league, row.home, row.away, row.season, row.odds_h) == (
        "E0",
        "Arsenal",
        "Man United",
        2026,
        1.55,
    )
    with pytest.raises(FixtureError, match="no league"):
        single_fixture("Arsenal", "Getafe", KNOWN, LEAGUES)
