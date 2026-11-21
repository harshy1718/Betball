"""Line-up strength: is today's starting XI stronger or weaker than this team's usual XI?

Player strength = Transfermarkt market value, the latest valuation published BEFORE the
match date (never one from after it). XI value = sum over the starting XI, scaled up to 11
when a few players have no valuation (at least 9 of 11 must have one).

The feature compares the XI with the team's own normal level, not with other teams: the
team ratings already know Man City is stronger than Burnley. What they cannot know is that
City rested six starters today. So:

    lineup = log(XI value today / median XI value in the team's league games of the
                 previous 365 days)       (needs 5+ such games; else 0 = "usual strength")

clipped to [-1.5, 0.5]. 0 means a usual XI, -0.3 means about 26% less value than usual.
Starting XIs are announced about an hour before kick-off, so this is pre-kick-off
information; using it means betting after line-ups are out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_VALUED = 9  # of the starting XI, players that must have a valuation
MIN_HISTORY = 5  # previous games needed for the team's usual level
WINDOW = "365D"
MAX_VALUATION_AGE = pd.Timedelta(days=730)
CLIP = (-1.5, 0.5)


def xi_values(lineups: pd.DataFrame, valuations: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """(game_id, club_id, date, xi_value, n_valued) for every starting XI in `games`."""
    starters = lineups[
        (lineups["type"] == "starting_lineup") & lineups["game_id"].isin(games["game_id"])
    ]
    starters = starters.drop_duplicates(["game_id", "club_id", "player_id"])
    starters = starters.merge(games[["game_id", "date"]], on="game_id")
    vals = valuations.dropna(subset=["date", "market_value_in_eur"])
    vals = vals.rename(columns={"date": "val_date"}).sort_values("val_date")
    starters = starters.sort_values("date")
    joined = pd.merge_asof(
        starters,
        vals,
        left_on="date",
        right_on="val_date",
        by="player_id",
        allow_exact_matches=False,  # strictly before the match day
        tolerance=MAX_VALUATION_AGE,
    )
    g = joined.groupby(["game_id", "club_id", "date"])
    out = g.agg(
        n_players=("player_id", "size"),
        n_valued=("market_value_in_eur", "count"),
        value=("market_value_in_eur", "sum"),
    ).reset_index()
    ok = (out["n_valued"] >= MIN_VALUED) & (out["n_players"] >= 10)
    out["xi_value"] = np.where(ok, out["value"] * 11 / out["n_valued"].clip(lower=1), np.nan)
    return out[["game_id", "club_id", "date", "xi_value", "n_valued"]]


def lineup_strength(xi: pd.DataFrame) -> pd.DataFrame:
    """Add `usual` (rolling median of earlier games) and `lineup` (the clipped log ratio)."""
    parts = []
    for _club, g in xi.sort_values("date").groupby("club_id"):
        s = g.set_index("date")["xi_value"]
        usual = s.rolling(WINDOW, min_periods=MIN_HISTORY, closed="left").median()
        parts.append(g.assign(usual=usual.to_numpy()))
    out = pd.concat(parts, ignore_index=True)
    ratio = np.log(out["xi_value"] / out["usual"])
    out["lineup"] = ratio.clip(*CLIP)
    return out


def trust_strength(lineups: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Price-free line-up strength from the coach's own choices (ADR 0007 amendment).

    For each player, his share of the team's league starts in the previous 365 days (the
    coach's revealed opinion of him). For today's XI:

        trust = log(sum of today's starters' shares / sum of the 11 highest shares)

    0 means the coach picked his usual XI; resting regulars makes it negative. Needs 5+
    earlier games; clipped like `lineup`. Market prices play no part, so inflated values
    for young or big-club players cannot distort it. Its blind spot is the opposite one:
    a new signing has no starts yet and counts as a non-regular.
    """
    starters = lineups[
        (lineups["type"] == "starting_lineup") & lineups["game_id"].isin(games["game_id"])
    ].drop_duplicates(["game_id", "club_id", "player_id"])
    starters = starters.merge(games[["game_id", "date"]], on="game_id")
    window = pd.Timedelta(WINDOW)
    rows: list[tuple[object, object, float]] = []
    for club, g in starters.groupby("club_id"):
        gm = g[["game_id", "date"]].drop_duplicates("game_id").sort_values("date")
        gm = gm.reset_index(drop=True)
        players = pd.Index(g["player_id"].unique())
        started = np.zeros((len(gm), len(players)))
        started[
            pd.Index(gm["game_id"]).get_indexer(g["game_id"]),
            players.get_indexer(g["player_id"]),
        ] = 1.0
        cum = np.vstack([np.zeros(len(players)), started.cumsum(axis=0)])
        dates = gm["date"].to_numpy()
        first = np.searchsorted(dates, dates - window, side="right")  # games within 365 days
        for i, (game_id, lo) in enumerate(zip(gm["game_id"], first, strict=True)):
            n = i - int(lo)
            value = np.nan
            if n >= MIN_HISTORY:
                share = (cum[i] - cum[lo]) / n
                top = np.sort(share)[-11:].sum()
                today = share[started[i] > 0].sum()
                if top > 0:
                    value = float(np.clip(np.log(max(today, 1e-6) / top), *CLIP))
            rows.append((game_id, club, value))
    return pd.DataFrame(rows, columns=["game_id", "club_id", "trust"])
