"""Golden bar per bet type and league (docs/adr/0014-golden-bar-by-bet-and-league.md)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.evaluation import window_search as ws
from football_edge.evaluation.window_both import leagues_in
from football_edge.markets.price_edge import bootstrap_roi

LEVEL = 1 - 0.05 / 8
LEAGUES = {"All three leagues": None, "Premier League": "E0", "Serie A": "I1", "La Liga": "SP1"}
MARKETS = {"1x2": "Match winner", "ou25": "Over/under 2.5"}


def _roi(b: pd.DataFrame) -> float:
    return float(b["profit"].sum() / b["stake"].sum()) if len(b) else float("nan")


def search(m: pd.DataFrame, market: str, league: str | None, seed: int = 0) -> dict[str, Any]:
    sub = m if league is None else m[m["league"] == league]
    disc = sub[sub["season"].between(*ws.DISCOVERY)]
    hold = sub[sub["season"].between(*ws.HOLDOUT)]
    ws.MIN_BETS = 200 if league is None else 100
    try:
        c_disc, c_hold = ws.candidates(disc, market), ws.candidates(hold, market)
        grid = ws.scan(c_disc).sort_values("roi", ascending=False)
        plc = ws.scan(ws.candidates(disc, market, np.random.default_rng(seed)))
    finally:
        ws.MIN_BETS = 200
    best = grid.iloc[0]
    b = ws._in(c_hold, best["lo"], best["hi"]).reset_index(drop=True)
    roi = _roi(b)
    lo, hi = bootstrap_roi(b, 2000, LEVEL) if len(b) >= 20 else (float("nan"), float("nan"))
    bar = ws._in(c_hold, 0.60, 0.64)
    return {
        "window": f"{best['lo']:.0%}-{best['hi']:.0%}", "disc_bets": int(best["bets"]),
        "disc_roi": float(best["roi"]), "hold_bets": len(b), "hold_roi": roi,
        "hold_won": float(b["won"].mean()) if len(b) else float("nan"),
        "lo": lo, "hi": hi, "found": len(b) > 0 and roi > 0 and lo > 0,
        "profitable": int((grid["roi"] > 0).sum()), "windows": len(grid),
        "placebo": int((plc["roi"] > 0).sum()), "bar_bets": len(bar), "bar_roi": _roi(bar),
    }  # fmt: skip


def run(m: pd.DataFrame, lo_min: float = 0.20) -> str:
    ws.LO_MIN = lo_min
    try:
        return _run(m, lo_min)
    finally:
        ws.LO_MIN = 0.20


def _run(m: pd.DataFrame, lo_min: float) -> str:
    title = "" if lo_min <= 0.2 else f" (windows from {lo_min:.0%} up)"
    lines = [
        f"# Golden bar per bet type and league{title}",
        "",
        "Rules pre-registered in docs/adr/0014-golden-bar-by-bet-and-league.md. Model only, Bet365 "
        "prices, flat 100. Search 2018/19-2022/23, test on 2023/24-2025/26 (unseen). "
        f"Interval {LEVEL:.3%}.",
    ]
    found = []
    for market, mname in MARKETS.items():
        lines += [
            "",
            f"## {mname}",
            "",
            "| League | Best window (search) | Search bets | Search ROI | Unseen bets | Unseen won "
            "| Unseen ROI | Interval | Profitable windows: real / placebo | Golden bar? |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for lname, lg in leagues_in(m).items():
            r = search(m, market, lg)
            if r["found"]:
                found.append(f"{mname}, {lname}: {r['window']}")
            lines.append(
                f"| {lname} | {r['window']} | {r['disc_bets']:,} | {r['disc_roi']:+.1%} "
                f"| {r['hold_bets']:,} | {r['hold_won']:.0%} | {r['hold_roi']:+.1%} "
                f"| {r['lo']:+.1%} to {r['hi']:+.1%} | {r['profitable']} / {r['placebo']} of "
                f"{r['windows']} | {'FOUND' if r['found'] else 'no'} |"
            )
        if market == "1x2":
            lines += ["", "Your current bar (60-64%) on the unseen seasons, by league:", "",
                      "| League | Bets | ROI |", "|---|---|---|"]  # fmt: skip
            for lname, lg in leagues_in(m).items():
                r = search(m, market, lg)
                lines.append(f"| {lname} | {r['bar_bets']} | {r['bar_roi']:+.1%} |")
    n_search = 2 * len(leagues_in(m))
    lines += ["", "## Verdict", "",
              ("Golden bars found: " + "; ".join(found)) if found
              else f"No golden bar found in any of the {n_search} searches."]  # fmt: skip
    return "\n".join(lines) + "\n"
