"""Phase 1 audit for the Football Edge raw CSVs (football-data.co.uk).

Usage: python3 -I audit.py <raw_dir> <out_dir>

Read-only on the raw data. Writes manifest.csv and coverage.csv to <out_dir>
and prints a compact report to stdout.
"""
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.simplefilter("ignore")

raw_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

# Column groups we care about (names as published by football-data.co.uk).
GROUPS = {
    "result": ["FTHG", "FTAG", "FTR"],
    "halftime": ["HTHG", "HTAG"],
    "shots": ["HS", "AS"],
    "sot": ["HST", "AST"],
    "odds1x2_pre_B365": ["B365H", "B365D", "B365A"],
    "odds1x2_pre_PS": ["PSH", "PSD", "PSA"],
    "odds1x2_pre_Avg": ["AvgH", "AvgD", "AvgA"],
    "odds1x2_close_B365": ["B365CH", "B365CD", "B365CA"],
    "odds1x2_close_PS": ["PSCH", "PSCD", "PSCA"],
    "odds1x2_close_Avg": ["AvgCH", "AvgCD", "AvgCA"],
    "ou25_pre_B365": ["B365>2.5", "B365<2.5"],
    "ou25_pre_PS": ["P>2.5", "P<2.5"],
    "ou25_pre_Avg": ["Avg>2.5", "Avg<2.5"],
    "ou25_close_B365": ["B365C>2.5", "B365C<2.5"],
    "ou25_close_PS": ["PC>2.5", "PC<2.5"],
    "ou25_close_Avg": ["AvgC>2.5", "AvgC<2.5"],
    # Older seasons used different column names for the same kind of odds.
    "legacy_1x2_BbAv": ["BbAvH", "BbAvD", "BbAvA"],
    "legacy_ou25_BbAv": ["BbAv>2.5", "BbAv<2.5"],
    "legacy_ou25_GB": ["GB>2.5", "GB<2.5"],
}


def read_one(path: Path) -> pd.DataFrame:
    # These files are sometimes latin-1 and carry trailing blank lines.
    # utf-8-sig strips the byte-order mark some files start with (it was hiding "Div").
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    df = df.dropna(how="all").copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "Date" in df.columns:
        df = df[df["Date"].notna()]
    return df


records = []   # per-file manifest rows
frames = []    # cleaned frames with season tagged

for path in sorted(raw_dir.glob("*.csv")):
    df = read_one(path)
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    bad_dates = int(df["Date_parsed"].isna().sum())
    dmin, dmax = df["Date_parsed"].min(), df["Date_parsed"].max()
    # Season start year: Aug-Dec -> that year, Jan-Jul -> previous year.
    season = dmin.year if dmin.month >= 7 else dmin.year - 1
    # League comes from the filename prefix (E0 / I1 / SP1), which is reliable.
    div = re.match(r"^(E0|I1|SP1)", path.name).group(1)
    df = df.assign(season=season, league=div, source_file=path.name)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna())
    rec = {
        "file": path.name, "league": div, "season": season,
        "rows": len(df), "teams": len(teams), "ncols": len(df.columns),
        "date_min": dmin.date(), "date_max": dmax.date(), "bad_dates": bad_dates,
    }
    for g, cols in GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if len(present) == len(cols):
            rec[g] = round(float(df[cols].notna().all(axis=1).mean()) * 100, 1)
        else:
            rec[g] = None  # columns not in file at all
    records.append(rec)
    frames.append(df)

manifest = pd.DataFrame(records).sort_values(["league", "season"]).reset_index(drop=True)
manifest.to_csv(out_dir / "manifest.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

print("=== FILE -> SEASON MAP, ROW COUNTS, DATE RANGE ===")
print(manifest[["file", "league", "season", "rows", "teams", "ncols",
                "date_min", "date_max", "bad_dates"]].to_string(index=False))

print("\n=== SEASONS PER LEAGUE (expect 12 distinct, 2014..2025) ===")
for lg, g in manifest.groupby("league"):
    seasons = sorted(g["season"].tolist())
    dup = [s for s in set(seasons) if seasons.count(s) > 1]
    print(lg, seasons, "DUPLICATE SEASONS:" if dup else "", dup if dup else "")

print("\n=== COVERAGE % (non-null across the full group) ===")
cov_cols = ["league", "season", "rows"] + list(GROUPS.keys())
cov = manifest[cov_cols]
cov.to_csv(out_dir / "coverage.csv", index=False)
print(cov.to_string(index=False))
print("(None/NaN = the columns do not exist in that file at all)")

all_df = pd.concat(frames, ignore_index=True)

print("\n=== INTEGRITY CHECKS ===")
print("total rows:", len(all_df))
# Required fields missing
for c in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date_parsed"]:
    n = int(all_df[c].isna().sum())
    print(f"missing {c}: {n}")
# FTR consistent with goals
exp = pd.Series("D", index=all_df.index)
exp[all_df["FTHG"] > all_df["FTAG"]] = "H"
exp[all_df["FTHG"] < all_df["FTAG"]] = "A"
incons = int((exp != all_df["FTR"]).sum())
print("FTR inconsistent with FTHG/FTAG:", incons)
# Duplicates
dups = all_df.duplicated(subset=["league", "Date_parsed", "HomeTeam", "AwayTeam"], keep=False)
print("duplicate (league,date,home,away) rows:", int(dups.sum()))
# Same team on both sides
print("home==away rows:", int((all_df["HomeTeam"] == all_df["AwayTeam"]).sum()))
# Out-of-season-window dates (Jul-start check)
months = all_df["Date_parsed"].dt.month
print("matches in Jun/Jul (e.g. COVID-delayed):",
      all_df.loc[months.isin([6, 7]), ["league", "season"]].value_counts().to_dict())
# Games per team per season (expect 38)
games = pd.concat([
    all_df[["league", "season", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
    all_df[["league", "season", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
]).value_counts().reset_index(name="n")
odd = games[games["n"] != 38]
print("team-seasons with games != 38:", len(odd))
if len(odd):
    print(odd.to_string(index=False))

print("\n=== TEAM NAMES: names NOT present in all 12 seasons (promotion/relegation or renames) ===")
for lg, g in all_df.groupby("league"):
    names = pd.concat([
        g[["season", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
        g[["season", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
    ]).drop_duplicates()
    by_team = names.groupby("team")["season"].apply(lambda s: sorted(s))
    print(f"\n{lg}: {len(by_team)} distinct names")
    for team, ss in by_team.items():
        if len(ss) < 12:
            print(f"  {team:<22} {len(ss):>2} seasons: {ss}")

print("\n=== GOALS SANITY (mean goals per match by league/season) ===")
gs = all_df.assign(tot=all_df["FTHG"] + all_df["FTAG"]).groupby(["league", "season"]).agg(
    mean_total=("tot", "mean"), home=("FTHG", "mean"), away=("FTAG", "mean"))
print(gs.round(2).to_string())
