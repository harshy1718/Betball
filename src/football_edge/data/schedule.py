"""Full-season schedule from football-data.org (free tier, current season).

One request per league: GET https://api.football-data.org/v4/competitions/{code}/matches
with header X-Auth-Token. The token is read from the FOOTBALL_DATA_ORG_KEY environment
variable, never from a file. Free tier: 10 requests a minute; we use 3 per refresh.

Their team names ("Manchester United FC", "Club Atlético de Madrid") are mapped to ours
("Man United", "Ath Madrid"): normalise (drop accents and words like FC/CF/AC/de), then
configs/team_aliases.yaml, then a unique word match against our names. Anything still
unmatched is reported so an alias can be added; it is never guessed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from football_edge.leagues import FOOTBALL_DATA_ORG
from football_edge.predict.fixtures import season_of

log = logging.getLogger(__name__)

API = "https://api.football-data.org/v4/competitions/{code}/matches"
CODES = FOOTBALL_DATA_ORG
TOKEN_ENV = "FOOTBALL_DATA_ORG_KEY"
_STOP = {
    "fc", "cf", "afc", "ac", "acf", "ssc", "ss", "as", "us", "sc", "cfc", "bc", "calcio",
    "club", "de", "del", "la", "futbol", "balompie", "rc", "rcd", "ca", "ud", "cd", "sad",
    "1907", "1909", "1913", "1919", "1936",
}  # fmt: skip


class ScheduleError(RuntimeError):
    """Raised when the schedule cannot be fetched or understood."""


def norm(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", " ", plain.replace("&", " ")).split()
    return " ".join(w for w in words if w not in _STOP and w != "and")


def build_name_map(
    api_names: set[str], known: dict[str, set[str]], league: str, aliases_path: Path | None
) -> tuple[dict[str, str], list[str]]:
    """API name -> our name for one league; plus the names that could not be matched."""
    ours = known.get(league, set())
    by_norm = {norm(n): n for n in ours}
    alias: dict[str, str] = {}
    if aliases_path is not None and aliases_path.exists():
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
        alias = {norm(k): v for k, v in (raw.get(league) or {}).items()}
    mapping, missing = {}, []
    for api in sorted(api_names):
        n = norm(api)
        found = by_norm.get(n) or alias.get(n)
        if found is None:  # unique word-subset match, e.g. "brighton" in "brighton hove albion"
            words = set(n.split())
            hits = [o for k, o in by_norm.items() if k and set(k.split()) <= words]
            found = hits[0] if len(hits) == 1 else None
        if found is None:
            missing.append(api)
        else:
            mapping[api] = found
    return mapping, missing


def http_get_json(url: str, token: str, timeout: float = 60.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"X-Auth-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read())
    return data


def parse_matches(payload: dict[str, Any], league: str) -> pd.DataFrame:
    """football-data.org matches JSON -> one row per match (team names as they give them)."""
    if "matches" not in payload:
        raise ScheduleError(f"{league}: unexpected answer: {str(payload)[:200]}")
    rows = []
    for m in payload["matches"]:
        refs = [r.get("name") for r in m.get("referees") or [] if r.get("type") == "REFEREE"]
        ft = (m.get("score") or {}).get("fullTime") or {}
        rows.append(
            {
                "league": league,
                "match_id": m["id"],
                "kickoff_utc": pd.Timestamp(m["utcDate"]),
                "matchday": m.get("matchday"),
                "status": m.get("status"),
                "home_api": (m.get("homeTeam") or {}).get("name"),
                "away_api": (m.get("awayTeam") or {}).get("name"),
                "goals_home": ft.get("home"),
                "goals_away": ft.get("away"),
                "referee": refs[0] if refs else None,
            }
        )
    return pd.DataFrame(rows)


def fetch_schedule(
    leagues: tuple[str, ...],
    known: dict[str, set[str]],
    aliases_path: Path | None,
    get: Callable[[str, str], dict[str, Any]] = http_get_json,
    token: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Whole current season for each league, with our team names. Returns (table, problems)."""
    token = token or os.environ.get(TOKEN_ENV, "")
    if not token:
        raise ScheduleError(
            f"set your football-data.org token in the {TOKEN_ENV} environment variable"
        )
    parts, problems = [], []
    for lg in leagues:
        code = CODES.get(lg)
        if code is None:
            continue
        df = parse_matches(get(API.format(code=code), token), lg)
        names = set(df["home_api"].dropna()) | set(df["away_api"].dropna())
        mapping, missing = build_name_map(names, known, lg, aliases_path)
        problems += [f"{lg}: no match for '{n}' (add it to configs/team_aliases.yaml)"
                     for n in missing]  # fmt: skip
        df["home"] = df["home_api"].map(mapping)
        df["away"] = df["away_api"].map(mapping)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    # Match dates in UK time, like football-data.co.uk dates.
    local = out["kickoff_utc"].dt.tz_convert("Europe/London")
    out["date"] = local.dt.tz_localize(None).dt.normalize()
    out["season"] = season_of(out["date"])
    return out.sort_values(["kickoff_utc", "league"]).reset_index(drop=True), problems


UPCOMING = ("SCHEDULED", "TIMED")


def upcoming(schedule: pd.DataFrame, today: pd.Timestamp, days: int) -> pd.DataFrame:
    """Matches still to be played from `today` through `today + days`, with known names."""
    end = today + pd.Timedelta(days=days)
    sel = (
        schedule["status"].isin(UPCOMING)
        & schedule["date"].between(today, end)
        & schedule["home"].notna()
        & schedule["away"].notna()
    )
    return schedule[sel].reset_index(drop=True)
