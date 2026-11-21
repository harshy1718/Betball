"""New-manager flag: is this team within its first few league games under a new manager?

    new_manager = 1 if the current manager had led the team in fewer than 6 of its
                  previous league games in a row, AND the team's earlier games were under
                  someone else (so the first games in our data are not mistaken for a change)
                  else 0

The manager for a match is known before kick-off. The famous "new manager bounce" is
largely regression to the mean (managers are sacked after bad runs, and bad runs end anyway),
and the team ratings already follow recent form, so this feature may well add nothing.
The pre-registered test (ADR 0007) decides.
"""

from __future__ import annotations

import pandas as pd

NEW_FOR_GAMES = 6


def new_manager_flags(games: pd.DataFrame) -> pd.DataFrame:
    """(game_id, club_id, new_manager) for both sides of every league game."""
    sides = pd.concat(
        [
            games[["game_id", "date", "home_club_id", "home_club_manager_name"]].set_axis(
                ["game_id", "date", "club_id", "manager"], axis=1
            ),
            games[["game_id", "date", "away_club_id", "away_club_manager_name"]].set_axis(
                ["game_id", "date", "club_id", "manager"], axis=1
            ),
        ]
    ).sort_values(["club_id", "date"])
    rows = []
    for club, g in sides.groupby("club_id", sort=False):
        prev: str | None = None
        run = 0  # consecutive previous games under the current manager
        seen_other = False
        for game_id, manager in zip(g["game_id"], g["manager"], strict=True):
            name = manager if isinstance(manager, str) and manager.strip() else None
            if name is None:
                rows.append((game_id, club, 0))
                continue
            if prev is not None and name != prev:
                seen_other, run = True, 0
            flag = int(seen_other and run < NEW_FOR_GAMES)
            rows.append((game_id, club, flag))
            run = run + 1 if name == prev or prev is None else 1
            prev = name
    return pd.DataFrame(rows, columns=["game_id", "club_id", "new_manager"])
