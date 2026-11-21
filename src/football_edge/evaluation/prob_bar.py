"""A probability bar per market (docs/adr/0011-probability-bar-test.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_edge.data.loader import read_raw_csv
from football_edge.markets.price_edge import bootstrap_roi

BARS = {
    "1x2": np.round(np.arange(0.40, 0.8501, 0.05), 2),
    "ou25": np.round(np.arange(0.50, 0.7001, 0.02), 2),
    "btts": np.round(np.arange(0.50, 0.6601, 0.02), 2),
}
PRICE_COLS = {
    "b365": ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
    "avg": ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
}


def load_prices(raw_dir: Path, leagues: list[str], seasons: range) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        for s in seasons:
            path = raw_dir / f"{lg}_{s}.csv"
            if not path.exists():
                continue
            df = read_raw_csv(path)
            t = pd.DataFrame({
                "league": lg,
                "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.normalize(),
                "home": df["HomeTeam"].str.strip(), "away": df["AwayTeam"].str.strip(),
            })  # fmt: skip
            for src, cols in PRICE_COLS.items():
                for name, col in zip(("h", "d", "a", "over", "under"), cols, strict=True):
                    v = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
                    t[f"{src}_{name}"] = v.where(v > 1.0) if isinstance(v, pd.Series) else v
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def picks(m: pd.DataFrame, market: str, src: str | None) -> pd.DataFrame:
    """Per match: the model's most likely outcome, its probability, whether it won, its price."""
    hg, ag = m["goals_home"].to_numpy(int), m["goals_away"].to_numpy(int)
    if market == "1x2":
        probs = m[["p_h", "p_d", "p_a"]].to_numpy(float)
        i = probs.argmax(axis=1)
        actual = np.select([hg > ag, hg == ag], [0, 1], 2)
        won = i == actual
        price = (m[[f"{src}_h", f"{src}_d", f"{src}_a"]].to_numpy(float)[np.arange(len(m)), i]
                 if src else np.nan)  # fmt: skip
        side = np.array(["home", "draw", "away"])[i]
        p = probs.max(axis=1)
    else:
        q = m["p_over25" if market == "ou25" else "p_btts"].to_numpy(float)
        yes = q >= 0.5
        happened = (hg + ag >= 3) if market == "ou25" else ((hg > 0) & (ag > 0))
        won = np.where(yes, happened, ~happened)
        p = np.where(yes, q, 1 - q)
        side = np.where(yes, "over" if market == "ou25" else "yes",
                        "under" if market == "ou25" else "no")  # fmt: skip
        price = (np.where(yes, m[f"{src}_over"], m[f"{src}_under"]).astype(float)
                 if src else np.nan)  # fmt: skip
    out = pd.DataFrame({"league": m["league"], "season": m["season"], "date": m["date"],
                        "home": m["home"], "away": m["away"], "side": side, "p": p,
                        "won": won, "odds": price})  # fmt: skip
    out["stake"] = 100.0
    out["profit"] = np.where(out["won"], 100.0 * (out["odds"] - 1), -100.0)
    return out


def _roi(b: pd.DataFrame) -> float:
    return float(b["profit"].sum() / b["stake"].sum()) if len(b) else float("nan")


def walk_forward(
    pk: pd.DataFrame, bars: np.ndarray
) -> tuple[pd.DataFrame, dict[int, float | None]]:
    pk = pk.dropna(subset=["odds"])
    chosen: dict[int, float | None] = {}
    bets = []
    for s in range(2020, 2026):
        past = pk[pk["season"] < s]
        best, best_roi = None, 0.0
        for bar in bars:
            b = past[past["p"] >= bar]
            if len(b) >= 100 and _roi(b) > best_roi:
                best, best_roi = float(bar), _roi(b)
        chosen[s] = best
        if best is not None:
            now = pk[pk["season"] == s]
            bets.append(now[now["p"] >= best])
    return (pd.concat(bets) if bets else pk.iloc[:0]), chosen


def run(preds: pd.DataFrame, raw_dir: Path, leagues: list[str]) -> str:
    prices = load_prices(raw_dir, leagues, range(2018, 2026))
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    m = p.merge(prices, on=["league", "date", "home", "away"], how="left")
    test = m["season"].between(2020, 2025)
    level = 0.975
    lines = [
        "# A probability bar per market (model only)",
        "",
        "Rules pre-registered in docs/adr/0011-probability-bar-test.md. "
        f"Test seasons 2020/21-2025/26 ({int(test.sum()):,} matches). "
        f"Flat 100 per bet. Intervals {level:.1%}.",
    ]
    names = {"1x2": "Match winner", "ou25": "Over/under 2.5"}
    for market in ("1x2", "ou25"):
        lines += ["", f"## {names[market]}", ""]
        for src, label in (("b365", "Bet365"), ("avg", "market average")):
            pk = picks(m, market, src)
            bets, chosen = walk_forward(pk, BARS[market])
            seasons = {str(s): _roi(g) for s, g in bets.groupby("season")}
            if len(bets):
                lo, hi = bootstrap_roi(bets.reset_index(drop=True), 2000, level)
                roi = _roi(bets)
                ok = roi > 0 and lo > 0 and sum(x > 0 for x in seasons.values()) >= 4
                verdict = "PROFITABLE" if ok else "NOT PROFITABLE"
                lines.append(
                    f"**Walk-forward, {label} prices:** {len(bets):,} bets, profit "
                    f"{bets['profit'].sum():+,.0f} on {bets['stake'].sum():,.0f} staked, "
                    f"ROI {roi:+.2%} "
                    f"(interval {lo:+.2%} to {hi:+.2%}), won {bets['won'].mean():.1%}, "
                    f"seasons in profit {sum(x > 0 for x in seasons.values())} of "
                    f"{len(seasons)}. **{verdict}.** Bars chosen: "
                    + ", ".join(
                        f"{s}: {'none' if b is None else f'{b:.2f}'}" for s, b in chosen.items()
                    )
                )
            else:
                lines.append(f"**Walk-forward, {label} prices:** no bar was ever profitable on "
                             "earlier seasons, so no bets. **NOT PROFITABLE.**")  # fmt: skip
            lines.append("")
        pk = picks(m[test], market, "b365").dropna(subset=["odds"])
        lines += ["Every bar on 2020-2025 at Bet365 prices (hindsight; information only):", "",
                  "| Bar | Bets | Won | Avg odds | ROI |", "|---|---|---|---|---|"]  # fmt: skip
        for bar in BARS[market]:
            b = pk[pk["p"] >= bar]
            if len(b):
                lines.append(f"| {bar:.2f} | {len(b):,} | {b['won'].mean():.1%} "
                             f"| {b['odds'].mean():.2f} | {_roi(b):+.2%} |")  # fmt: skip
    pk = picks(m[test], "btts", None)
    lines += ["", "## Both teams to score (no historical prices: profit cannot be measured)", "",
              "| Bar | Picks | Came true | Break-even price |", "|---|---|---|---|"]  # fmt: skip
    for bar in BARS["btts"]:
        b = pk[pk["p"] >= bar]
        if len(b):
            lines.append(f"| {bar:.2f} | {len(b):,} | {b['won'].mean():.1%} "
                         f"| {1 / b['won'].mean():.2f} |")  # fmt: skip
    return "\n".join(lines) + "\n"
