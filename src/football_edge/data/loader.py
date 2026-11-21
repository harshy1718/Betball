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
_KEEP = (
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",  # corners
    "AC",
    "HY",  # yellow cards
    "AY",
    "HR",  # red cards
    "AR",
)


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


def pick_odds(
    df: pd.DataFrame,
    priority: tuple[OddsSource, ...],
    names: tuple[str, ...] = ("h", "d", "a"),
) -> pd.DataFrame:
    """Per row, take the first source with all its prices; record which one was used."""
    out = pd.DataFrame({n: float("nan") for n in names}, index=df.index)
    out["src"] = pd.Series([None] * len(df), index=df.index, dtype="object")
    for source in priority:
        if len(source.cols) != len(names) or not all(c in df.columns for c in source.cols):
            continue
        # Prices of 1.0 or less are placeholders in the source files (e.g. 0.0): treat as missing.
        prices = df[list(source.cols)].apply(pd.to_numeric, errors="coerce")
        prices = prices.where(prices > 1.0)
        ok = prices.notna().all(axis=1) & out[names[0]].isna()
        out.loc[ok, list(names)] = prices.loc[ok].to_numpy(dtype=float)
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
    for prefix, priority in (("max", cfg.max_pre_priority), ("maxc", cfg.max_close_priority)):
        best = pick_odds(df, priority)
        t[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]] = best[["h", "d", "a"]]
    ou = pick_odds(df, cfg.ou_close_priority, names=("over", "under"))
    t[["ou_close_over", "ou_close_under"]] = ou[["over", "under"]]
    t["ou_close_src"] = ou["src"]
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
    """Copy raw files into data/raw (season-coded), write manifest, build processed table.

    Several source files can hold the same league-season (an older download and a newer
    one). The one with the most matches is used: a season file only ever grows, so the
    biggest is the most recent. Ties go to the most recently modified file.
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    best: dict[tuple[str, int], tuple[Path, int]] = {}
    for path in sorted(src_dir.glob("*.csv")):
        if not _FILE_RE.match(path.name):
            continue
        parsed = parse_file(path, cfg)
        key = (str(parsed["league"].iat[0]), int(str(parsed["season"].iat[0])))
        current = best.get(key)
        if (
            current is None
            or len(parsed) > current[1]
            or (len(parsed) == current[1] and path.stat().st_mtime > current[0].stat().st_mtime)
        ):
            best[key] = (path, len(parsed))
    manifest: list[dict[str, object]] = []
    for (league, season), (path, rows) in sorted(best.items()):
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
                "rows": rows,
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


def load_matches(cfg: Config, features: bool = False) -> pd.DataFrame:
    """Read the processed table (what models and evaluation should always use).

    With `features`, the pre-kick-off match features from `build-features`
    (data/processed/features.parquet) are joined on; unknown values stay NaN.
    """
    df = pd.read_parquet(cfg.matches_path)
    validate_matches(df)
    if features:
        path = cfg.processed_dir / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `football-edge build-features` first")
        keys = ["league", "date", "home", "away"]
        feats = pd.read_parquet(path)
        df = df.merge(feats, on=keys, how="left", validate="one_to_one")
        if len(feats) != len(df) or feats[keys].isna().any().any():
            log.warning("features were built from a different matches table: rebuild them")
    return df


def needs_features(params: dict[str, object]) -> bool:
    """True if a model's parameters ask for external match features."""
    return bool(params.get("covariates"))
