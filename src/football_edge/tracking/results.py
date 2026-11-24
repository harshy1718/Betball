"""Final results for logged predictions: find the played match and settle any leg."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.trader.legs import count_test, goal_test

COUNT_MARKETS = ("corners", "cards")


def count_total(row: Any, market: str) -> float | None:
    """Total corners, or total cards (yellows + reds, the way the count models count)."""
    cols = ("HC", "AC") if market == "corners" else ("HY", "AY", "HR", "AR")
    total = 0.0
    for c in cols:
        v: Any = row.get(c) if hasattr(row, "get") else getattr(row, c, None)
        if v is None or pd.isna(v):
            return None
        total += float(v)
    return total


def leg_won(row: Any, market: str, sel: str, line: float | None) -> bool | None:
    """Did this leg come in? None when the result lacks the needed stat."""
    if market in COUNT_MARKETS:
        tot = count_total(row, market)
        if tot is None or line is None:
            return None
        return bool(count_test(sel, float(line))(np.array([tot]))[0])
    h, a = np.array([int(row["FTHG"])]), np.array([int(row["FTAG"])])
    return bool(goal_test(market, sel, line)(h, a)[0])


def parse_line(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
        return None
    return float(v)


class Results:
    """Played matches, looked up by league, our team names and kickoff (within a day)."""

    def __init__(self, matches: pd.DataFrame) -> None:
        m = matches.dropna(subset=["FTHG", "FTAG"])
        self._by: dict[tuple[str, str, str], list[tuple[pd.Timestamp, pd.Series]]] = {}
        for _, r in m.iterrows():
            self._by.setdefault((str(r["league"]), str(r["home"]), str(r["away"])), []).append(
                (pd.Timestamp(r["date"]), r)
            )

    def find(self, league: str, home: str, away: str, kickoff_utc: str) -> pd.Series | None:
        day = pd.Timestamp(kickoff_utc)
        day = (day.tz_convert(None) if day.tzinfo else day).normalize()
        for date, r in self._by.get((league, home, away), []):
            if abs(date - day) <= pd.Timedelta(days=1):
                return r
        return None
