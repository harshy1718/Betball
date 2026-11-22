"""The golden bar: back the model's match-winner pick when its probability is 60%-64%.

Found by the window search (reports/window_search, ADR 0012) and NOT proven: the search's own pick
(61-64%) lost on unseen seasons, and 60-64% was chosen after seeing them. Kept by Harsh's choice
(22 Nov 2026) and tracked forward: every qualifying pick is logged, and settled when results arrive.
Judge it at 200 settled picks, not before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLS = ["logged_on", "league", "date", "home", "away", "pick", "prob", "site_odds",
        "result", "won", "profit"]  # fmt: skip
JUDGE_AT = 200


BARS = (  # (config key under `staking`, default log file, scoreboard name)
    ("golden_bar", "bets/golden_bar.csv", "Match-winner bar"),
    ("golden_ou", "bets/golden_ou.csv", "Over/under bar"),
    ("golden_cs", "bets/golden_cs.csv", "Correct-score bar"),
)
CS_MIN_ODDS = 6.80  # break-even of the correct-score bar in 2020-26 (14.7% came true)


def winner_window(gb: dict[str, Any], league: str) -> tuple[float, float]:
    """The match-winner window for one league: `by_league` override, else lo/hi."""
    w = (gb.get("by_league") or {}).get(league)
    if w:
        return float(w[0]), float(w[1])
    return float(gb.get("lo", 0.60)), float(gb.get("hi", 0.64))


def _pct(lo: float, hi: float) -> str:
    return f"{lo:.0%}-{hi:.0%}"


def golden_picks(fc: pd.DataFrame, gb: dict[str, Any]) -> pd.DataFrame:
    """Match-winner bar: fixtures whose model favourite is inside its league's window."""
    probs = fc[["p_h", "p_d", "p_a"]].to_numpy(float)
    i = probs.argmax(axis=1)
    p = probs.max(axis=1)
    odds_cols = ["odds_h", "odds_d", "odds_a"]
    odds = (fc[odds_cols].to_numpy(float)[np.arange(len(fc)), i]
            if all(c in fc for c in odds_cols) else np.full(len(fc), np.nan))  # fmt: skip
    side = np.array(["home", "draw", "away"])[i]
    names = np.where(side == "home", fc["home"], np.where(side == "away", fc["away"], "draw"))
    out = pd.DataFrame({
        "league": fc["league"].to_numpy(), "date": pd.to_datetime(fc["date"]).to_numpy(),
        "home": fc["home"].to_numpy(), "away": fc["away"].to_numpy(), "pick": side,
        "label": [f"{n} win" if s != "draw" else "draw" for n, s in zip(names, side, strict=True)],
        "prob": p, "site_odds": odds,
    })  # fmt: skip
    windows = [winner_window(gb, str(lg)) for lg in out["league"]]
    keep = [lo <= q < hi for q, (lo, hi) in zip(out["prob"], windows, strict=True)]
    return out[keep].reset_index(drop=True)


def ou_picks(fc: pd.DataFrame, go: dict[str, Any]) -> pd.DataFrame:
    """Over/under bar: the model's over/under 2.5 pick, in the listed leagues, inside [lo, hi)."""
    q = fc["p_over25"].to_numpy(float)
    over = q >= 0.5
    p = np.where(over, q, 1 - q)
    if "odds_over25" in fc and "odds_under25" in fc:
        odds = np.where(over, fc["odds_over25"], fc["odds_under25"]).astype(float)
    else:
        odds = np.full(len(fc), np.nan)
    out = pd.DataFrame({
        "league": fc["league"].to_numpy(), "date": pd.to_datetime(fc["date"]).to_numpy(),
        "home": fc["home"].to_numpy(), "away": fc["away"].to_numpy(),
        "pick": np.where(over, "over", "under"),
        "label": np.where(over, "over 2.5 goals", "under 2.5 goals"), "prob": p, "site_odds": odds,
    })  # fmt: skip
    lo, hi = float(go.get("lo", 0.51)), float(go.get("hi", 0.56))
    leagues = set(go.get("leagues") or [])
    keep = (out["prob"] >= lo) & (out["prob"] < hi)
    if leagues:
        keep &= out["league"].isin(leagues)
    return out[keep].reset_index(drop=True)


def flags(f: pd.Series, st: dict[str, Any]) -> dict[str, str]:
    """Golden-bar text for one forecast row: keys 'win', 'ou', 'cs' ('' when not golden).
    Independent of price: a golden pick is golden whatever the odds."""
    row = f.to_frame().T
    out = {"win": "", "ou": "", "cs": ""}
    gb, go, cs = (st.get(k) or {} for k in ("golden_bar", "golden_ou", "golden_cs"))
    if gb:
        g = golden_picks(row, gb)
        if len(g):
            lo, hi = winner_window(gb, str(f["league"]))
            out["win"] = (
                f"⭐ {g.iloc[0]['label']} {float(g.iloc[0]['prob']):.0%} (bar {_pct(lo, hi)})"
            )
    if go and "p_over25" in f:
        g = ou_picks(row, go)
        if len(g):
            out["ou"] = f"⭐ {g.iloc[0]['label']} {float(g.iloc[0]['prob']):.0%}"
    if cs and "top_score_p" in f and float(f["top_score_p"]) >= float(cs.get("lo", 0.12)):
        out["cs"] = (f"⭐ {f['top_score']} {float(f['top_score_p']):.1%} "
                     f"(take at {CS_MIN_ODDS:.2f}+)")  # fmt: skip
    return out


def cs_picks(fc: pd.DataFrame, lo: float) -> pd.DataFrame:
    """Correct-score golden bar: fixtures whose single likeliest score has probability >= lo."""
    out = pd.DataFrame({
        "league": fc["league"].to_numpy(), "date": pd.to_datetime(fc["date"]).to_numpy(),
        "home": fc["home"].to_numpy(), "away": fc["away"].to_numpy(),
        "pick": fc["top_score"].to_numpy(), "label": fc["top_score"].to_numpy(),
        "prob": fc["top_score_p"].to_numpy(float), "site_odds": np.nan,
    })  # fmt: skip
    return out[out["prob"] >= lo].reset_index(drop=True)


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    log = pd.read_csv(path, parse_dates=["date", "logged_on"])
    return log.astype({"result": object, "won": object})


def log_picks(picks: pd.DataFrame, path: Path, today: pd.Timestamp) -> int:
    """Append new picks (one per match); a later run only fills in a missing price."""
    log = read(path)
    keys = ["league", "date", "home", "away"]
    new = 0
    for r in picks.to_dict("records"):
        date, odds = pd.Timestamp(r["date"]), float(r["site_odds"])
        hit = ((log["league"] == r["league"]) & (log["date"] == date)
               & (log["home"] == r["home"]) & (log["away"] == r["away"]))  # fmt: skip
        if hit.any():
            if log.loc[hit, "site_odds"].isna().all() and not np.isnan(odds):
                log.loc[hit, "site_odds"] = odds
            continue
        row = {"logged_on": today, "league": r["league"], "date": date, "home": r["home"],
               "away": r["away"], "pick": r["pick"], "prob": round(float(r["prob"]), 4),
               "site_odds": odds, "result": np.nan, "won": np.nan, "profit": np.nan}  # fmt: skip
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        new += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    log.sort_values(keys).to_csv(path, index=False, float_format="%.4f")
    return new


def set_price(path: Path, team: str, odds: float) -> list[str]:
    """Record your site's price on unsettled picks whose home or away team contains `team`
    (case-insensitive). Returns the matches updated."""
    log = read(path)
    if log.empty:
        return []
    t = team.lower()
    hit = log["result"].isna() & (
        log["home"].str.lower().str.contains(t, regex=False)
        | log["away"].str.lower().str.contains(t, regex=False)
    )
    log.loc[hit, "site_odds"] = float(odds)
    log.to_csv(path, index=False, float_format="%.4f")
    return [f"{h} v {a}: {p} at {odds:.2f}"
            for h, a, p in zip(log.loc[hit, "home"], log.loc[hit, "away"], log.loc[hit, "pick"],
                               strict=True)]  # fmt: skip


def settle(path: Path, matches: pd.DataFrame, stake: float = 100.0) -> pd.DataFrame:
    log = read(path)
    if log.empty:
        return log
    res = {
        (str(lg), pd.Timestamp(d).normalize(), str(h), str(a)): (int(hg), int(ag))
        for lg, d, h, a, hg, ag in zip(
            matches["league"], matches["date"], matches["home"], matches["away"],
            matches["FTHG"], matches["FTAG"], strict=True,
        )
    }  # fmt: skip
    dates = pd.to_datetime(log["date"]).dt.normalize()
    prices = pd.to_numeric(log["site_odds"], errors="coerce")
    for k in log.index[log["result"].isna()]:
        key = (str(log.at[k, "league"]), dates[k], str(log.at[k, "home"]), str(log.at[k, "away"]))
        if key not in res:
            continue
        hg, ag = res[key]
        pick = str(log.at[k, "pick"])
        if "-" in pick:  # correct score, e.g. "2-1"
            won = pick == f"{hg}-{ag}"
        elif pick in ("over", "under"):
            won = (hg + ag >= 3) == (pick == "over")
        else:
            won = pick == ("home" if hg > ag else ("draw" if hg == ag else "away"))
        log.at[k, "result"] = f"{hg}-{ag}"
        log.at[k, "won"] = won
        price = float(prices[k])
        log.at[k, "profit"] = (
            np.nan if np.isnan(price) else (stake * (price - 1) if won else -stake)
        )
    log.to_csv(path, index=False, float_format="%.4f")
    return log


def scoreboard(log: pd.DataFrame, name: str = "Golden bar") -> str:
    done = log[log["result"].notna()]
    priced = done[done["profit"].notna()]
    if done.empty:
        return f"{name}: {len(log)} picks logged, none settled yet (judge at {JUDGE_AT})."
    won = done["won"].astype(bool).mean()
    breakeven = 1 / won if won else float("inf")
    text = (
        f"{name}: {len(done)} of {JUDGE_AT} settled picks needed; won {won:.1%} (model expected "
        f"{done['prob'].astype(float).mean():.1%}; break-even price {breakeven:.2f}). "
    )
    if len(priced):
        roi = priced["profit"].sum() / (100.0 * len(priced))
        text += (f"{len(priced)} with a price: profit {priced['profit'].sum():+,.0f} per 100 "
                 f"staked, ROI {roi:+.1%}. ")  # fmt: skip
    return text + f"{len(log) - len(done)} waiting for results."
