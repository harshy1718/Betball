"""The bet log: every bet placed, settled automatically from real results.

One CSV (bets/log.csv, kept out of git). Columns:
  placed_on, kind (single|double), legs, odds, stake, prob, result, profit
`legs` is one or more "league|YYYY-MM-DD|home|away|market|selection" joined by " + ".
`result` is blank until settled, then win / loss / void.

The log is what makes the monthly budget and stop-loss real, and it is the only
honest scoreboard: probabilities, prices taken and money won or lost, side by side.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_edge.betting.staking import MonthStatus, leg_key

LOG_COLS = ["placed_on", "kind", "legs", "odds", "stake", "prob", "result", "profit"]


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(path, dtype={"result": "string"}, keep_default_na=False, na_values=[""])
    missing = [c for c in LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"bet log {path} is missing columns: {missing}")
    df["placed_on"] = pd.to_datetime(df["placed_on"])
    return df[LOG_COLS]


def write_log(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["placed_on"] = pd.to_datetime(out["placed_on"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, float_format="%.4f")


def month_status(log: pd.DataFrame, today: pd.Timestamp) -> MonthStatus:
    if log.empty:
        return MonthStatus(0.0, 0.0)
    this_month = log[log["placed_on"].dt.to_period("M") == today.to_period("M")]
    settled = this_month["result"].notna()
    return MonthStatus(
        settled_profit=float(this_month.loc[settled, "profit"].sum()),
        pending_stake=float(this_month.loc[~settled, "stake"].sum()),
    )


def new_entries(singles: pd.DataFrame, doubles: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Log rows for every staked recommendation (prices must be known to log a bet)."""
    rows = []
    for _, s in singles[(singles["stake"] > 0) & singles["odds"].notna()].iterrows():
        rows.append(["single", leg_key(s), s["odds"], s["stake"], s["prob"]])
    for _, d in doubles[doubles["stake"] > 0].iterrows():
        rows.append(["double", f"{d['leg1']} + {d['leg2']}", d["odds"], d["stake"], d["prob"]])
    out = pd.DataFrame(rows, columns=["kind", "legs", "odds", "stake", "prob"])
    out.insert(0, "placed_on", today.normalize())
    out["result"] = pd.NA
    out["profit"] = float("nan")
    return out[LOG_COLS]


def _leg_result(leg: str, results: dict[tuple[str, str, str, str], tuple[int, int]]) -> str | None:
    league, date, home, away, market, selection = leg.split("|")
    score = results.get((league, date, home, away))
    if score is None:
        return None  # not played yet, or not in our data yet
    h, a = score
    won = {
        ("1x2", "home"): h > a,
        ("1x2", "draw"): h == a,
        ("1x2", "away"): h < a,
        ("ou25", "over"): h + a >= 3,
        ("ou25", "under"): h + a < 3,
        ("btts", "yes"): h > 0 and a > 0,
        ("btts", "no"): h == 0 or a == 0,
    }.get((market, selection))
    if won is None:
        raise ValueError(f"cannot settle market/selection {market}/{selection}")
    return "win" if won else "loss"


def settle(log: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill result and profit for every unsettled bet whose matches have been played."""
    results: dict[tuple[str, str, str, str], tuple[int, int]] = {
        (str(lg), f"{d:%Y-%m-%d}", str(h), str(a)): (int(hg), int(ag))
        for lg, d, h, a, hg, ag in zip(
            matches["league"],
            matches["date"],
            matches["home"],
            matches["away"],
            matches["FTHG"].astype(int),
            matches["FTAG"].astype(int),
            strict=True,
        )
    }
    out = log.copy()
    n = 0
    for i, row in out[out["result"].isna()].iterrows():
        legs = [_leg_result(leg, results) for leg in str(row["legs"]).split(" + ")]
        if any(r is None for r in legs):
            continue
        res = "win" if all(r == "win" for r in legs) else "loss"
        out.at[i, "result"] = res
        out.at[i, "profit"] = row["stake"] * (row["odds"] - 1) if res == "win" else -row["stake"]
        n += 1
    return out, n


def summary(log: pd.DataFrame) -> pd.DataFrame:
    """Per month: bets, staked, profit, return on stake, and wins expected vs actual."""
    done = log[log["result"].isin(["win", "loss"])].copy()
    if done.empty:
        return pd.DataFrame()
    done["month"] = done["placed_on"].dt.to_period("M").astype(str)
    g = done.groupby("month")
    out = pd.DataFrame(
        {
            "bets": g.size(),
            "staked": g["stake"].sum(),
            "profit": g["profit"].sum(),
            "wins": g["result"].apply(lambda r: int((r == "win").sum())),
            "wins_expected": g["prob"].sum(),  # if the model is calibrated, close to wins
        }
    )
    out["return"] = out["profit"] / out["staked"]
    return out
