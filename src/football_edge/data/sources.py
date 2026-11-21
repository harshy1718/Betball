"""Download the latest files from football-data.co.uk (free, no key).

* Current-season results per league, e.g. https://www.football-data.co.uk/mmz4281/2627/E0.csv
  (results, shots, corners, cards and odds; updated after each round).
* Upcoming fixtures with pre-match odds for all leagues: .../fixtures.csv

Downloads go into the raw source folder under a dated name ("E0_2026_20261122.csv"), so
earlier downloads are never overwritten. `ingest` then uses, per league-season, the file
with the most matches. A download is refused if it does not parse, has the wrong league,
or has fewer matches than the copy we already have (a season file only ever grows).
"""

from __future__ import annotations

import io
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk"
Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "football-edge/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
    return data


def season_code(season: int) -> str:
    """2026 (= 2026/27) -> '2627'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _rows(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", encoding_errors="replace")
    df = df.dropna(how="all")
    return df[df["Date"].notna()] if "Date" in df.columns else df.iloc[0:0]


@dataclass(frozen=True)
class Download:
    league: str
    path: Path | None  # None when refused
    rows: int
    note: str


def download_results(
    src_dir: Path,
    season: int,
    leagues: tuple[str, ...],
    existing_rows: dict[str, int],
    fetch: Fetch = http_get,
    stamp: str | None = None,
) -> list[Download]:
    """Fetch the current-season file for each league into `src_dir` (validated)."""
    stamp = stamp or pd.Timestamp.today().strftime("%Y%m%d")
    src_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for lg in leagues:
        url = f"{BASE}/mmz4281/{season_code(season)}/{lg}.csv"
        try:
            data = fetch(url)
            df = _rows(data)
        except Exception as e:  # network, HTTP or parse error: keep what we have
            out.append(Download(lg, None, 0, f"download failed: {e}"))
            continue
        problems = []
        if not {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(df.columns):
            problems.append("missing columns")
        elif set(df["Div"].dropna().unique()) != {lg}:
            problems.append(f"wrong league {sorted(df['Div'].dropna().unique())}")
        have = existing_rows.get(lg, 0)
        if len(df) < have:
            problems.append(f"{len(df)} matches, fewer than the {have} we already have")
        if problems:
            out.append(Download(lg, None, len(df), "refused: " + "; ".join(problems)))
            continue
        path = src_dir / f"{lg}_{season}_{stamp}.csv"
        path.write_bytes(data)
        new = len(df) - have
        out.append(Download(lg, path, len(df), f"{len(df)} matches ({new:+d} new)"))
    return out


def download_fixtures(dest: Path, fetch: Fetch = http_get) -> int:
    """Upcoming fixtures with pre-match odds (all leagues) -> `dest`. Returns row count."""
    data = fetch(f"{BASE}/fixtures.csv")
    df = _rows(data)
    if not {"Div", "Date", "HomeTeam", "AwayTeam"} <= set(df.columns):
        raise ValueError("fixtures.csv has an unexpected format")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(df)
