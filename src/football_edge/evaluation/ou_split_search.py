"""Over/under windows by side and league (docs/adr/0013-ou-window-search-split.md)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.evaluation import window_search as ws
from football_edge.markets.price_edge import bootstrap_roi

LEVEL = 1 - 0.05 / 8
LEAGUES = {"all": None, "Premier League": "E0", "Serie A": "I1", "La Liga": "SP1"}


def _side(c: pd.DataFrame, side: str) -> pd.DataFrame:
    # candidates() stacks [over, under] per match, so even rows are over and odd rows under.
    return c.iloc[0::2] if side == "over" else c.iloc[1::2]


def run(m: pd.DataFrame, seed: int = 0) -> str:
    rows: list[tuple[str, str, dict[str, Any] | None]] = []
    for side in ("over", "under"):
        for lg_name, lg in LEAGUES.items():
            sub = m if lg is None else m[m["league"] == lg]
            disc = sub[sub["season"].between(*ws.DISCOVERY)]
            hold = sub[sub["season"].between(*ws.HOLDOUT)]
            ws.MIN_BETS = 200 if lg is None else 100
            c_disc = _side(ws.candidates(disc, "ou25"), side)
            c_hold = _side(ws.candidates(hold, "ou25"), side)
            grid = ws.scan(c_disc).sort_values("roi", ascending=False)
            plc = ws.scan(_side(ws.candidates(disc, "ou25", np.random.default_rng(seed)), side))
            if grid.empty:
                rows.append((side, lg_name, None))
                continue
            best = grid.iloc[0]
            b = ws._in(c_hold, best["lo"], best["hi"]).reset_index(drop=True)
            roi = float(b["profit"].sum() / b["stake"].sum()) if len(b) else float("nan")
            lo, hi = bootstrap_roi(b, 2000, LEVEL) if len(b) >= 20 else (float("nan"),) * 2
            rows.append((side, lg_name, {
                "window": f"{best['lo']:.0%}-{best['hi']:.0%}", "disc_bets": int(best["bets"]),
                "disc_roi": float(best["roi"]), "hold_bets": len(b), "hold_roi": roi,
                "lo": lo, "hi": hi, "ok": len(b) > 0 and roi > 0 and lo > 0,
                "profitable": int((grid["roi"] > 0).sum()), "windows": len(grid),
                "placebo": int((plc["roi"] > 0).sum()),
            }))  # fmt: skip
    ws.MIN_BETS = 200
    lines = [
        "# Over/under 2.5 windows by side and league (final search)",
        "",
        "Rules pre-registered in docs/adr/0013-ou-window-search-split.md. Model only, Bet365 "
        f"prices, flat 100. Search 2018/19-2022/23, test 2023/24-2025/26. Interval {LEVEL:.3%}.",
        "",
        "| Side | League | Best window (search) | Search bets | Search ROI | Unseen bets "
        "| Unseen ROI | Interval | Profitable windows: real / placebo | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for side, lg_name, r in rows:
        if r is None:
            lines.append(f"| {side} | {lg_name} | none with enough bets | - | - | - | - | - | - "
                         "| DOES NOT |")  # fmt: skip
            continue
        lines.append(
            f"| {side} | {lg_name} | {r['window']} | {r['disc_bets']:,} | {r['disc_roi']:+.1%} "
            f"| {r['hold_bets']:,} | {r['hold_roi']:+.1%} | {r['lo']:+.1%} to {r['hi']:+.1%} "
            f"| {r['profitable']} / {r['placebo']} of {r['windows']} "
            f"| {'MAKES MONEY' if r['ok'] else 'DOES NOT'} |"
        )
    return "\n".join(lines) + "\n"
