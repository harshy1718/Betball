"""Every window scored in BOTH periods (search 2018/19-2022/23 and unseen 2023/24-2025/26); the
best 2 per bet type and league are the windows that made money in both, ranked by the weaker of
the two returns. Exploratory: the unseen seasons are used for choosing here, so these windows are
candidates for forward tracking, not proven edges. A placebo (model probabilities shuffled)
shows how many 'profitable in both' windows luck alone produces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.evaluation import window_search as ws

LEAGUES = {"All three leagues": None, "Premier League": "E0", "Serie A": "I1", "La Liga": "SP1"}
MARKETS = {"1x2": "Match winner", "ou25": "Over/under 2.5"}


def _scan_all(c: pd.DataFrame) -> pd.DataFrame:
    """Bets and ROI of every window (no minimum)."""
    old = ws.MIN_BETS
    ws.MIN_BETS = 1
    try:
        return ws.scan(c)
    finally:
        ws.MIN_BETS = old


def both(m: pd.DataFrame, market: str, league: str | None,
         rng: np.random.Generator | None = None) -> pd.DataFrame:  # fmt: skip
    sub = m if league is None else m[m["league"] == league]
    disc = sub[sub["season"].between(*ws.DISCOVERY)]
    hold = sub[sub["season"].between(*ws.HOLDOUT)]
    a = _scan_all(ws.candidates(disc, market, rng)).rename(
        columns={"bets": "bets_s", "roi": "roi_s"}
    )
    b = _scan_all(ws.candidates(hold, market, rng)).rename(
        columns={"bets": "bets_u", "roi": "roi_u"}
    )
    t = a.merge(b, on=["lo", "hi"], how="inner")
    need = 200 if league is None else 100
    t = t[(t["bets_s"] >= need) & (t["bets_u"] >= need * 0.6)]  # unseen has 3 seasons, not 5
    t["worse"] = t[["roi_s", "roi_u"]].min(axis=1)
    t["all_roi"] = (t["roi_s"] * t["bets_s"] + t["roi_u"] * t["bets_u"]) / (
        t["bets_s"] + t["bets_u"]
    )
    return t.sort_values("worse", ascending=False).reset_index(drop=True)


def run(m: pd.DataFrame, seed: int = 0, lo_min: float = 0.20) -> str:
    ws.LO_MIN = lo_min
    try:
        return _run(m, seed, lo_min)
    finally:
        ws.LO_MIN = 0.20


def _run(m: pd.DataFrame, seed: int, lo_min: float) -> str:
    title = "" if lo_min <= 0.2 else f" (windows from {lo_min:.0%} up)"
    lines = [
        f"# Every window in both periods: the 2 best per bet type and league{title}",
        "",
        "Each window is scored on the search seasons (2018/19-2022/23) AND the unseen seasons "
        "(2023/24-2025/26); model only, Bet365 prices, flat 100. 'Best' = profitable in both, "
        "ranked by the weaker of the two returns. Both periods are used for choosing, so these "
        "are candidates to track, not proven. 'Luck control' = the same count with the model's "
        "probabilities shuffled (a model that knows nothing).",
    ]
    for market, mname in MARKETS.items():
        lines += ["", f"## {mname}", "",
                  "| League | Rank | Window | Bets (search) | Return (search) | Bets (unseen) "
                  "| Return (unseen) | Return, all 8 seasons | Windows profitable in both: "
                  "real / luck control |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for lname, lg in LEAGUES.items():
            t = both(m, market, lg)
            plc = both(m, market, lg, np.random.default_rng(seed))
            n_real, n_plc = int((t["worse"] > 0).sum()), int((plc["worse"] > 0).sum())
            top = t[t["worse"] > 0].head(2)
            if top.empty:
                lines.append(f"| {lname} | - | none profitable in both | - | - | - | - | - "
                             f"| {n_real} / {n_plc} of {len(t)} |")  # fmt: skip
                continue
            for k, r in enumerate(top.to_dict("records"), start=1):
                lines.append(
                    f"| {lname if k == 1 else ''} | {k} | {r['lo']:.0%}-{r['hi']:.0%} "
                    f"| {int(r['bets_s']):,} | {r['roi_s']:+.1%} | {int(r['bets_u']):,} "
                    f"| {r['roi_u']:+.1%} | {r['all_roi']:+.1%} "
                    f"| {f'{n_real} / {n_plc} of {len(t)}' if k == 1 else ''} |"
                )
    return "\n".join(lines) + "\n"
