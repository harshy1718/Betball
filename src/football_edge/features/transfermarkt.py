"""Read the Transfermarkt dataset and link its clubs and games to our matches.

Source: github.com/dcaribou/transfermarkt-datasets (CC0 licence). Download the zip
(transfermarkt-datasets.zip) and extract it; point `build-features --tm` at the folder.
Only three tables are used: games, game_lineups, player_valuations.

Linking: our data (football-data.co.uk) and Transfermarkt name clubs differently
("Man United" vs "Manchester United Football Club"). Instead of a hand-made list, the
link is learned from the data: matches on the same date in the same league with the same
score are almost always the same match, so for each of our team names the Transfermarkt
club that keeps turning up in its place is that team. Weak or clashing links are refused.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from football_edge.predict.fixtures import season_of

log = logging.getLogger(__name__)

# Transfermarkt competition id -> our league code.
COMPETITIONS = {"GB1": "E0", "IT1": "I1", "ES1": "SP1"}

_COLS = {
    "games": [
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
    "game_lineups": ["game_id", "club_id", "player_id", "type"],
    "player_valuations": ["player_id", "date", "market_value_in_eur"],
}


class LinkError(ValueError):
    """Raised when our teams cannot be linked to Transfermarkt clubs reliably."""


def _find(folder: Path, name: str) -> Path:
    for pattern in (f"{name}.csv.gz", f"{name}.csv"):
        hits = sorted(folder.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name}.csv(.gz) not found under {folder}")


def read_table(folder: Path, name: str) -> pd.DataFrame:
    path = _find(folder, name)
    df = pd.read_csv(path, usecols=_COLS[name], low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log.info("read %s: %d rows", path.name, len(df))
    return df


def league_games(games: pd.DataFrame) -> pd.DataFrame:
    """Games of our three leagues, with our league code and season."""
    g = games[games["competition_id"].isin(COMPETITIONS)].copy()
    g = g.dropna(subset=["date", "home_club_goals", "away_club_goals"])
    g["league"] = g["competition_id"].map(COMPETITIONS)
    g["season"] = season_of(g["date"])
    for c in ("home_club_goals", "away_club_goals"):
        g[c] = g[c].astype(int)
    return g.reset_index(drop=True)


def link_teams(
    matches: pd.DataFrame, games: pd.DataFrame, min_share: float = 0.6, min_matches: int = 10
) -> pd.DataFrame:
    """Return (league, team, club_id, share, n): our team name -> Transfermarkt club."""
    j = matches.merge(
        games,
        left_on=["league", "date", "FTHG", "FTAG"],
        right_on=["league", "date", "home_club_goals", "away_club_goals"],
    )
    pairs = pd.concat(
        [
            j[["league", "home", "home_club_id"]].set_axis(["league", "team", "club_id"], axis=1),
            j[["league", "away", "away_club_id"]].set_axis(["league", "team", "club_id"], axis=1),
        ]
    )
    counts = pairs.value_counts().rename("count").reset_index()
    totals = counts.groupby(["league", "team"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    counts["n"] = totals
    best = counts.sort_values("count", ascending=False).drop_duplicates(["league", "team"])
    weak = best[(best["share"] < min_share) | (best["n"] < min_matches)]
    if not weak.empty:
        log.warning("teams not linked (too little agreement):\n%s", weak.to_string(index=False))
    best = best.drop(weak.index)
    clash = best[best.duplicated(["league", "club_id"], keep=False)]
    if not clash.empty:
        raise LinkError(f"two of our teams link to one club:\n{clash.to_string(index=False)}")
    return best[["league", "team", "club_id", "share", "n"]].reset_index(drop=True)


def link_games(
    matches: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame, max_days: int = 3
) -> pd.Series:
    """Transfermarkt game_id for each of our matches (NaN where none is found).

    Same league, same home and away clubs, and the closest date within `max_days`
    (dates can differ by a day between sources when a match is moved or listed in UTC).
    """
    ids = teams.set_index(["league", "team"])["club_id"]
    m = matches[["league", "date", "home", "away"]].copy()
    m["home_id"] = ids.reindex(pd.MultiIndex.from_frame(m[["league", "home"]])).to_numpy()
    m["away_id"] = ids.reindex(pd.MultiIndex.from_frame(m[["league", "away"]])).to_numpy()
    g = games[["league", "home_club_id", "away_club_id", "game_id", "date"]].rename(
        columns={"home_club_id": "home_id", "away_club_id": "away_id", "date": "tm_date"}
    )
    j = m.reset_index().merge(g, on=["league", "home_id", "away_id"])
    j["gap"] = (j["date"] - j["tm_date"]).abs()
    j = j[j["gap"] <= pd.Timedelta(days=max_days)].sort_values("gap")
    best = j.drop_duplicates("index").set_index("index")["game_id"]
    return best.reindex(matches.index)
