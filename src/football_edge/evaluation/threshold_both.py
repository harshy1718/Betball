"""Open-ended bars ("bet every pick at X% or more", X = 50%..90%), scored in both periods
(2018/19-2022/23 and 2023/24-2025/26); the best 2 per bet type and league (profitable in both,
ranked by the weaker return) are then checked on 2026/27 so far, which nothing has seen.
Model only, Bet365 pre-match prices, flat 100."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.evaluation import window_search as ws
from football_edge.evaluation.window_both import leagues_in

LEAGUES = {"All three leagues": None, "Premier League": "E0", "Serie A": "I1", "La Liga": "SP1"}
MARKETS = {"1x2": "Match winner", "ou25": "Over/under 2.5"}
BARS = np.round(np.arange(0.50, 0.9001, 0.01), 2)


def _score(c: pd.DataFrame, bar: float) -> tuple[int, float]:
    b = c[c["p"] >= bar]
    if not len(b):
        return 0, float("nan")
    return len(b), float(b["profit"].sum() / (100.0 * len(b)))


def table(m: pd.DataFrame, market: str, league: str | None,
          rng: np.random.Generator | None = None) -> pd.DataFrame:  # fmt: skip
    sub = m if league is None else m[m["league"] == league]
    c_s = ws.candidates(sub[sub["season"].between(*ws.DISCOVERY)], market, rng)
    c_u = ws.candidates(sub[sub["season"].between(*ws.HOLDOUT)], market, rng)
    need = 200 if league is None else 100
    rows = []
    for bar in BARS:
        ns, rs = _score(c_s, bar)
        nu, ru = _score(c_u, bar)
        if ns >= need and nu >= need * 0.6:
            rows.append({"bar": bar, "bets_s": ns, "roi_s": rs, "bets_u": nu, "roi_u": ru})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["worse"] = t[["roi_s", "roi_u"]].min(axis=1)
    return t.sort_values("worse", ascending=False).reset_index(drop=True)


def run(m: pd.DataFrame, m26: pd.DataFrame | None, seed: int = 0) -> str:
    lines = [
        "# Open-ended bars: bet every pick at X% or more",
        "",
        "Every bar from 50% to 90% (1% steps), scored on 2018/19-2022/23 and 2023/24-2025/26; "
        "best 2 = profitable in both, ranked by the weaker return. Then checked on 2026/27 so far "
        "(never used for choosing). Model only, Bet365 prices, flat 100.",
    ]
    for market, mname in MARKETS.items():
        lines += ["", f"## {mname}", "",
                  "| League | Rank | Bar | Bets 2018-22 | Return | Bets 2023-26 | Return "
                  "| Bets 2026/27 | Return 2026/27 | Bars profitable in both: real / luck |",
                  "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for lname, lg in leagues_in(m).items():
            t = table(m, market, lg)
            plc = table(m, market, lg, np.random.default_rng(seed))
            n_real = int((t["worse"] > 0).sum()) if len(t) else 0
            n_plc = int((plc["worse"] > 0).sum()) if len(plc) else 0
            top = t[t["worse"] > 0].head(2) if len(t) else t
            luck = f"{n_real} / {n_plc} of {len(t)}"
            if top.empty:
                lines.append(f"| {lname} | - | none profitable in both | - | - | - | - | - | - "
                             f"| {luck} |")  # fmt: skip
                continue
            for k, r in enumerate(top.to_dict("records"), start=1):
                n26, r26 = (0, float("nan"))
                if m26 is not None:
                    sub26 = m26 if lg is None else m26[m26["league"] == lg]
                    n26, r26 = _score(ws.candidates(sub26, market), float(r["bar"]))
                lines.append(
                    f"| {lname if k == 1 else ''} | {k} | {r['bar']:.0%}+ | {int(r['bets_s']):,} "
                    f"| {r['roi_s']:+.1%} | {int(r['bets_u']):,} | {r['roi_u']:+.1%} | {n26} "
                    f"| {r26:+.1%} | {luck if k == 1 else ''} |"
                )
    return "\n".join(lines) + "\n"
