"""Ingestion: raw football-data.co.uk CSVs -> validated, season-tagged matches table.

Raw files are immutable inputs. `ingest` copies them to data/raw under season-coded
names with SHA-256 hashes in a manifest, so any later result is traceable to exact bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from football_edge.config import Config, OddsSource
from football_edge.data.schema import validate_matches

log = logging.getLogger(__name__)
_FILE_RE = re.compile(r"^(E0|I1|SP1)")
# Results and in-match stats. These describe a match AFTER it is played: models may
# learn from them for past matches but must never see them for the match predicted.
_KEEP = ("FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HS", "AS", "HST", "AST")


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read one raw file. utf-8-sig strips the BOM some files carry."""
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df[df["Date"].notna()].copy()


def infer_season(dates: pd.Series) -> int:
    """Season start year from the earliest date (Jul-Dec -> that year, else previous)."""
    first = dates.min()
    return int(first.year if first.month >= 7 else first.year - 1)


def pick_odds(df: pd.DataFrame, priority: tuple[OddsSource, ...]) -> pd.DataFrame:
    """Per row, take the first source with all three prices; record which one was used."""
    out = pd.DataFrame(
        {"h": float("nan"), "d": float("nan"), "a": float("nan"), "src": None}, index=df.index
    )
    out["src"] = out["src"].astype("object")
    for source in priority:
        if not all(c in df.columns for c in source.cols):
            continue
        ok = df[list(source.cols)].notna().all(axis=1) & out["h"].isna()
        out.loc[ok, ["h", "d", "a"]] = df.loc[ok, list(source.cols)].to_numpy(dtype=float)
        out.loc[ok, "src"] = source.name
    return out


def parse_file(path: Path, cfg: Config) -> pd.DataFrame:
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognised file name: {path.name}")
    df = read_raw_csv(path)
    date = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    close = pick_odds(df, cfg.closing_priority)
    pre = pick_odds(df, cfg.pre_priority)
    t = pd.DataFrame(
        {
            "league": m.group(1),
            "season": infer_season(date),
            "date": date,
            "home": df["HomeTeam"].str.strip(),
            "away": df["AwayTeam"].str.strip(),
        }
    )
    for c in _KEEP:
        t[c] = df[c] if c in df.columns else pd.NA
    t[["close_h", "close_d", "close_a"]] = close[["h", "d", "a"]]
    t["close_src"] = close["src"]
    t[["pre_h", "pre_d", "pre_a"]] = pre[["h", "d", "a"]]
    t["pre_src"] = pre["src"]
    return t


def build_matches(files: list[Path], cfg: Config) -> pd.DataFrame:
    frames = [parse_file(p, cfg) for p in sorted(files)]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "league", "home"]).reset_index(drop=True)
    validate_matches(out)
    log.info("built %d matches across %d files", len(out), len(frames))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(src_dir: Path, cfg: Config) -> pd.DataFrame:
    """Copy raw files into data/raw (season-coded), write manifest, build processed table."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for path in sorted(src_dir.glob("*.csv")):
        if not _FILE_RE.match(path.name):
            continue
        parsed = parse_file(path, cfg)
        league, season = str(parsed["league"].iat[0]), int(str(parsed["season"].iat[0]))
        dest = cfg.raw_dir / f"{league}_{season}.csv"
        if dest.exists() and _sha256(dest) != _sha256(path):
            # Partial current-season file legitimately changes; archived seasons must not.
            log.warning("raw file changed since last ingest: %s", dest.name)
        shutil.copyfile(path, dest)
        manifest.append(
            {
                "file": dest.name,
                "league": league,
                "season": season,
                "rows": len(parsed),
                "sha256": _sha256(dest),
                "original_name": path.name,
            }
        )
    (cfg.raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = [cfg.raw_dir / str(m["file"]) for m in manifest]
    matches = build_matches(files, cfg)
    matches.to_parquet(cfg.matches_path, index=False)
    log.info("wrote %s", cfg.matches_path)
    return matches


def load_matches(cfg: Config) -> pd.DataFrame:
    """Read the processed table (what models and evaluation should always use)."""
    df = pd.read_parquet(cfg.matches_path)
    validate_matches(df)
    return df
