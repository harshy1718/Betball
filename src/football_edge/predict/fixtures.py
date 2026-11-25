"""Read upcoming fixtures (and the prices on offer) into one standard table.

Two input formats are accepted:

1. football-data.co.uk `fixtures.csv` (has a `Div` column). Team names already match
   our history. Prices come from one bookmaker column set, chosen by `odds_source`.
2. A simple CSV typed by hand from the betting site you use:
       league,date,home,away,odds_h,odds_d,odds_a,odds_over25,odds_under25,odds_btts_yes,odds_btts_no
   Every odds column is optional. Team names may be the site's own; they are mapped
   through configs/team_aliases.yaml.

   Optional reference prices from a sharp bookmaker (Pinnacle), typed the same way:
       ref_h,ref_d,ref_a,ref_over25,ref_under25,ref_btts_yes,ref_btts_no
   Without them, the site's own prices are used as the market's view.

Output columns: league, season, date, home, away, the odds_* columns (prices you can bet at)
and the ref_* columns (prices that set the market's fair probability); NaN = not given.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd
import yaml

ODDS_COLS = (
    "odds_h",
    "odds_d",
    "odds_a",
    "odds_over25",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
)

_LEAGUE_ALIASES = {
    "e0": "E0",
    "epl": "E0",
    "premier league": "E0",
    "i1": "I1",
    "serie a": "I1",
    "sp1": "SP1",
    "la liga": "SP1",
    "laliga": "SP1",
    "d1": "D1",
    "bundesliga": "D1",
    "f1": "F1",
    "ligue 1": "F1",
    "ligue1": "F1",
}

REF_COLS = tuple(c.replace("odds_", "ref_") for c in ODDS_COLS)

# football-data column names per bookmaker: 1X2 prefix, over/under prefix.
_FD_SOURCES = {
    "Avg": ("Avg", "Avg"),
    "Max": ("Max", "Max"),
    "B365": ("B365", "B365"),
    "PS": ("PS", "P"),
}


class FixtureError(ValueError):
    """Raised when the fixtures file cannot be used as it is."""


def season_of(dates: pd.Series) -> pd.Series:
    """Season start year per date: July-December -> that year, else the previous year."""
    years = dates.dt.year
    out: pd.Series = years.where(dates.dt.month >= 7, years - 1).astype(int)
    return out


def _from_football_data(raw: pd.DataFrame, odds_source: str) -> pd.DataFrame:
    if odds_source not in _FD_SOURCES:
        raise FixtureError(f"odds_source must be one of {sorted(_FD_SOURCES)}")
    p1x2, pou = _FD_SOURCES[odds_source]
    out = pd.DataFrame(
        {
            "league": raw["Div"],
            "date": raw["Date"],
            "home": raw["HomeTeam"],
            "away": raw["AwayTeam"],
        }
    )
    cols = {
        "odds_h": f"{p1x2}H",
        "odds_d": f"{p1x2}D",
        "odds_a": f"{p1x2}A",
        "odds_over25": f"{pou}>2.5",
        "odds_under25": f"{pou}<2.5",
    }
    for ours, theirs in cols.items():
        out[ours] = raw[theirs] if theirs in raw.columns else float("nan")
    # Reference (market view): Pinnacle, the sharpest book; market average where missing.
    for ours, pin, avg in (
        ("ref_h", "PSH", "AvgH"),
        ("ref_d", "PSD", "AvgD"),
        ("ref_a", "PSA", "AvgA"),
        ("ref_over25", "P>2.5", "Avg>2.5"),
        ("ref_under25", "P<2.5", "Avg<2.5"),
    ):
        first = raw[pin] if pin in raw.columns else pd.Series(float("nan"), index=raw.index)
        second = raw[avg] if avg in raw.columns else pd.Series(float("nan"), index=raw.index)
        out[ours] = pd.to_numeric(first, errors="coerce").where(
            pd.to_numeric(first, errors="coerce") > 1.0, pd.to_numeric(second, errors="coerce")
        )
    return out


def _load_aliases(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {lg: {k.lower(): v for k, v in (m or {}).items()} for lg, m in raw.items()}


def resolve_names(
    fixtures: pd.DataFrame,
    known: dict[str, set[str]],
    aliases: dict[str, dict[str, str]],
    allow_unknown: bool = False,
) -> pd.DataFrame:
    """Map site names to history names; fail loudly on names we cannot place.

    A wrong name silently becomes a "new team" with an average rating, which gives
    confident-looking but meaningless probabilities. So unknown names stop the run
    unless `allow_unknown` (for a team genuinely new to our data).
    """
    out = fixtures.copy()
    problems: list[str] = []
    for side in ("home", "away"):
        fixed = []
        for lg, name in zip(out["league"], out[side], strict=True):
            names = known.get(lg, set())
            lookup = {n.lower(): n for n in names}
            key = name.lower()
            resolved = lookup.get(key) or aliases.get(lg, {}).get(key) or name
            if resolved not in names and not allow_unknown:
                hint = difflib.get_close_matches(name, sorted(names), n=3, cutoff=0.5)
                problems.append(f"{lg} '{name}' (closest: {', '.join(hint) or 'none'})")
            fixed.append(resolved)
        out[side] = fixed
    if problems:
        raise FixtureError(
            "unknown team names - add them to configs/team_aliases.yaml, or pass "
            "--allow-unknown if the team is new to our data:\n  " + "\n  ".join(problems)
        )
    return out


def load_fixtures(
    path: Path,
    known: dict[str, set[str]],
    leagues: tuple[str, ...],
    odds_source: str = "Avg",
    aliases_path: Path | None = None,
    allow_unknown: bool = False,
) -> pd.DataFrame:
    """Read a fixtures file in either format and return the standard table."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Div" in raw.columns:
        df = _from_football_data(raw, odds_source)
    else:
        missing = [c for c in ("league", "date", "home", "away") if c not in raw.columns]
        if missing:
            raise FixtureError(f"fixtures file is missing columns: {missing}")
        df = raw.copy()
    for c in ODDS_COLS + REF_COLS:
        if c not in df.columns:
            df[c] = float("nan")  # no price given for this market
    df["league"] = [
        _LEAGUE_ALIASES.get(str(x).strip().lower(), str(x).strip()) for x in df["league"]
    ]
    df = df[df["league"].isin(leagues)].copy()
    if df.empty:
        raise FixtureError(f"no fixtures for our leagues {list(leagues)} in {path.name}")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    for c in ODDS_COLS + REF_COLS:
        prices = pd.to_numeric(df[c], errors="coerce")
        df[c] = prices.where(prices > 1.0)  # 1.0 or less = placeholder, not a price
    df = resolve_names(df, known, _load_aliases(aliases_path), allow_unknown)
    if (df["home"] == df["away"]).any():
        raise FixtureError("a fixture has the same home and away team")
    df["season"] = season_of(df["date"])
    cols = ["league", "season", "date", "home", "away", *ODDS_COLS, *REF_COLS]
    return df[cols].sort_values(["date", "league", "home"]).reset_index(drop=True)


def single_fixture(
    home: str,
    away: str,
    known: dict[str, set[str]],
    leagues: tuple[str, ...],
    league: str | None = None,
    date: str | None = None,
    odds_1x2: tuple[float, float, float] | None = None,
    odds_ou25: tuple[float, float] | None = None,
    aliases_path: Path | None = None,
) -> pd.DataFrame:
    """One fixture typed on the command line. The league is found from the team names
    when not given; a name that fits no league (or two) stops with suggestions."""
    aliases = _load_aliases(aliases_path)
    when = pd.Timestamp(date) if date else pd.Timestamp.today().normalize()
    if league:
        candidates = [_LEAGUE_ALIASES.get(league.strip().lower(), league.strip())]
    else:
        candidates = []
        for lg in leagues:
            lookup = {n.lower(): n for n in known.get(lg, set())}
            if all(
                (n.lower() in lookup) or (n.lower() in aliases.get(lg, {})) for n in (home, away)
            ):
                candidates.append(lg)
        if len(candidates) != 1:
            hints = {
                n: difflib.get_close_matches(
                    n, sorted(set().union(*known.values())), n=3, cutoff=0.5
                )
                for n in (home, away)
            }
            found = "two leagues" if candidates else "no league"
            raise FixtureError(
                f"the two teams fit {found}; check the names or pass --league. "
                f"Closest names: {hints}"
            )
    row: dict[str, object] = {"league": candidates[0], "date": when, "home": home, "away": away}
    for c in ODDS_COLS + REF_COLS:
        row[c] = float("nan")
    if odds_1x2:
        row["odds_h"], row["odds_d"], row["odds_a"] = odds_1x2
    if odds_ou25:
        row["odds_over25"], row["odds_under25"] = odds_ou25
    df = pd.DataFrame([row])
    df = resolve_names(df, known, aliases)
    if (df["home"] == df["away"]).any():
        raise FixtureError("home and away are the same team")
    df["season"] = season_of(df["date"])
    return df[["league", "season", "date", "home", "away", *ODDS_COLS, *REF_COLS]]


def from_schedule(upcoming: pd.DataFrame, priced: pd.DataFrame | None) -> pd.DataFrame:
    """Scheduled matches as fixtures; prices joined from fixtures.csv where it has them."""
    df = upcoming[["league", "season", "date", "home", "away"]].copy()
    cols = list(ODDS_COLS + REF_COLS)
    if priced is not None and not priced.empty:
        keys = ["league", "date", "home", "away"]
        have = [c for c in cols if c in priced.columns]
        df = df.merge(priced[keys + have], on=keys, how="left")
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    return df[["league", "season", "date", "home", "away", *cols]].reset_index(drop=True)
