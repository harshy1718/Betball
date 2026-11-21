import numpy as np
import pandas as pd
import pytest

from football_edge.features.build import build_features, coverage
from football_edge.features.lineup import lineup_strength, xi_values
from football_edge.features.manager import new_manager_flags
from football_edge.features.transfermarkt import LinkError, league_games, link_teams

TEAMS = {"Arsenal": 11, "Chelsea": 631, "Spurs": 148, "Everton": 29}


@pytest.fixture()
def world():
    """Our matches and the same games in Transfermarkt form, over 2 seasons."""
    rng = np.random.default_rng(0)
    names = list(TEAMS)
    ours, tm = [], []
    day = pd.Timestamp("2023-08-12")
    gid = 1000
    for rnd in range(24):
        d = day + pd.Timedelta(days=7 * rnd)
        order = rng.permutation(names)
        for h, a in [(order[0], order[1]), (order[2], order[3])]:
            hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
            ours.append(("E0", 2023, d, h, a, hg, ag))
            mgr = {n: f"Boss {n}" for n in names}
            if rnd >= 12:
                mgr["Chelsea"] = "New Boss"  # Chelsea change manager from round 12
            tm.append(
                (gid, "GB1", d, TEAMS[h], TEAMS[a], hg, ag, mgr[h], mgr[a], f"{h} FC", f"{a} FC")
            )
            gid += 1
    matches = pd.DataFrame(
        ours, columns=["league", "season", "date", "home", "away", "FTHG", "FTAG"]
    )
    games = pd.DataFrame(
        tm,
        columns=[
            "game_id",
            "competition_id",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
            "home_club_manager_name",
            "away_club_manager_name",
            "home_club_name",
            "away_club_name",
        ],
    )
    # Every club fields the same 11 players, each valued 10m, except in round 20
    # Arsenal rest 6 starters for 6 players valued 1m.
    lineups, vals = [], []
    for g in games.itertuples():
        for club in (g.home_club_id, g.away_club_id):
            rested = club == 11 and g.date == day + pd.Timedelta(days=7 * 20)
            for k in range(11):
                pid = club * 100 + k + (50 if rested and k < 6 else 0)
                lineups.append((g.game_id, club, pid, "starting_lineup"))
            lineups.append((g.game_id, club, club * 100 + 99, "substitutes"))
    for club in TEAMS.values():
        for k in list(range(11)) + list(range(50, 56)):
            value = 1e6 if k >= 50 else 10e6
            vals.append((club * 100 + k, pd.Timestamp("2023-07-01"), value))
    lineups = pd.DataFrame(lineups, columns=["game_id", "club_id", "player_id", "type"])
    vals = pd.DataFrame(vals, columns=["player_id", "date", "market_value_in_eur"])
    return matches, games, lineups, vals


def test_teams_link_from_shared_results(world):
    matches, games, *_ = world
    links = link_teams(matches, league_games(games))
    assert dict(zip(links.team, links.club_id, strict=True)) == TEAMS


def test_link_refuses_two_teams_on_one_club(world):
    matches, games, *_ = world
    games = league_games(games).replace({"home_club_id": {29: 148}, "away_club_id": {29: 148}})
    with pytest.raises(LinkError, match="two of our teams"):
        link_teams(matches, games)


def test_unlinkable_team_is_left_out(world):
    matches, games, *_ = world
    games = league_games(games)
    games = games[(games.home_club_id != 29) & (games.away_club_id != 29)]  # no Everton games
    links = link_teams(matches, games)
    assert "Everton" not in set(links.team) and len(links) == 3


def test_rested_xi_shows_up_and_usual_level_uses_only_earlier_games(world):
    _, games, lineups, vals = world
    g = league_games(games)
    lu = lineup_strength(xi_values(lineups, vals, g))
    ars = lu[lu.club_id == 11].sort_values("date").reset_index(drop=True)
    assert ars["lineup"].iloc[:5].isna().all()  # fewer than 5 earlier games: unknown
    rested = ars[ars.xi_value < 100e6]
    assert len(rested) == 1
    assert rested["lineup"].iloc[0] == pytest.approx(np.log(56e6 / 110e6))
    normal = ars[(ars.xi_value == 110e6) & ars.lineup.notna()]
    assert (normal["lineup"] == 0).all()


def test_valuation_on_match_day_is_not_used(world):
    _, games, lineups, vals = world
    g = league_games(games)
    first_day = g["date"].min()
    late = vals.assign(date=first_day)  # published on the match day itself
    xi = xi_values(lineups, late, g)
    assert xi.loc[xi.date == first_day, "xi_value"].isna().all()
    assert xi.loc[xi.date > first_day, "xi_value"].notna().all()


def test_new_manager_flag(world):
    _, games, *_ = world
    g = league_games(games)
    flags = new_manager_flags(g).merge(g[["game_id", "date"]], on="game_id")
    chelsea = flags[flags.club_id == 631].sort_values("date")["new_manager"].tolist()
    change = chelsea.index(1)
    assert chelsea[:change] == [0] * change  # first games in the data are not a "change"
    assert chelsea[change : change + 6] == [1] * 6 and chelsea[change + 6 :] == [0] * len(
        chelsea[change + 6 :]
    )
    assert flags[flags.club_id != 631]["new_manager"].sum() == 0


def test_build_features_end_to_end(world):
    matches, games, lineups, vals = world
    feats, links = build_features(matches, games, lineups, vals)
    assert len(links) == 4
    assert len(feats) == len(matches) and feats["game_id"].notna().all()
    assert {"lineup_home", "lineup_away", "new_mgr_home", "new_mgr_away"} <= set(feats.columns)
    cov = coverage(feats, matches)
    assert cov.loc[2023, "linked_game"] == 1.0
    assert 0 < cov.loc[2023, "lineup_both"] < 1


def test_trust_sees_rested_regulars_without_prices(world):
    _, games, lineups, _ = world
    g = league_games(games)
    from football_edge.features.lineup import trust_strength

    t = trust_strength(lineups, g).merge(g[["game_id", "date"]], on="game_id")
    ars = t[t.club_id == 11].sort_values("date").reset_index(drop=True)
    assert ars["trust"].iloc[:5].isna().all()
    rested_day = g["date"].min() + pd.Timedelta(days=7 * 20)
    assert ars.loc[ars.date == rested_day, "trust"].iloc[0] == pytest.approx(np.log(5 / 11))
    assert (ars.loc[ars.trust.notna() & (ars.date < rested_day), "trust"] == 0).all()
