"""Build data/processed/features.parquet: one row per match in our table.

Columns: league, date, home, away, game_id, lineup_home, lineup_away, trust_home,
trust_away, new_mgr_home,
new_mgr_away, xi_home, xi_away (raw XI values in EUR, for inspection).
Missing values mean "unknown" and are treated as 0 (usual strength / no change) by models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from football_edge.features.lineup import lineup_strength, trust_strength, xi_values
from football_edge.features.manager import new_manager_flags
from football_edge.features.transfermarkt import league_games, link_games, link_teams, read_table

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "lineup_home",
    "lineup_away",
    "trust_home",
    "trust_away",
    "new_mgr_home",
    "new_mgr_away",
]
KEYS = ["league", "date", "home", "away"]


def build_features(
    matches: pd.DataFrame,
    games_raw: pd.DataFrame,
    lineups: pd.DataFrame,
    valuations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, team links)."""
    games = league_games(games_raw)
    teams = link_teams(matches, games)
    game_id = link_games(matches, games, teams)

    ids = teams.set_index(["league", "team"])["club_id"]
    out = matches[KEYS].copy()
    out["game_id"] = game_id
    out["home_id"] = ids.reindex(pd.MultiIndex.from_frame(matches[["league", "home"]])).to_numpy()
    out["away_id"] = ids.reindex(pd.MultiIndex.from_frame(matches[["league", "away"]])).to_numpy()

    lu = lineup_strength(xi_values(lineups, valuations, games))
    lu = lu.set_index(["game_id", "club_id"])
    trust = trust_strength(lineups, games).set_index(["game_id", "club_id"])["trust"]
    mgr = new_manager_flags(games).set_index(["game_id", "club_id"])["new_manager"]
    for side in ("home", "away"):
        key = pd.MultiIndex.from_arrays([out["game_id"], out[f"{side}_id"]])
        out[f"lineup_{side}"] = lu["lineup"].reindex(key).to_numpy()
        out[f"xi_{side}"] = lu["xi_value"].reindex(key).to_numpy()
        out[f"trust_{side}"] = trust.reindex(key).to_numpy()
        out[f"new_mgr_{side}"] = mgr.reindex(key).to_numpy()
    return out.drop(columns=["home_id", "away_id"]), teams


def coverage(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Share of matches per season with each feature known for both sides."""
    f = pd.DataFrame(
        {
            "season": matches["season"].to_numpy(),
            "linked_game": features["game_id"].notna().to_numpy(),
            "lineup_both": (
                features["lineup_home"].notna() & features["lineup_away"].notna()
            ).to_numpy(),
            "trust_both": (
                features["trust_home"].notna() & features["trust_away"].notna()
            ).to_numpy(),
            "manager_both": (
                features["new_mgr_home"].notna() & features["new_mgr_away"].notna()
            ).to_numpy(),
            "new_mgr_share": (
                (features["new_mgr_home"].fillna(0) + features["new_mgr_away"].fillna(0)) / 2
            ).to_numpy(),
        }
    )
    out = f.groupby("season").mean()
    out.insert(0, "matches", f.groupby("season").size())
    return out


def build_from_folder(matches: pd.DataFrame, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_features(
        matches,
        read_table(folder, "games"),
        read_table(folder, "game_lineups"),
        read_table(folder, "player_valuations"),
    )
