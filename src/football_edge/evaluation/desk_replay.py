"""Replay the trader desk on past seasons with nothing from the future (ADR 0020 follow-up).

The audit replay (24 Nov 2026) used confidence labels built from the very seasons it replayed,
and a corners/cards lift measured on 2020-2026. Here every input is walk-forward:
  * chances: the calibrated walk-forward predictions (each season's calibration fitted only
    on earlier seasons), or the raw ones when calibration is off;
  * confidence labels and skill flags for season s: built only from seasons 2019..s-1;
  * the corners/cards lift for a match: only matches played before that week's Monday;
  * the desk's own code picks every single and parlay (LegFactory, Book, Desk.best_parlay).
Every slip is graded from the final score and card totals. Intervals: 95% block bootstrap by
league x week (slips of the same match and week are not independent evidence).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd

from football_edge.models.calibration import adjust_draws, block_bootstrap
from football_edge.models.dixon_coles import score_grid
from football_edge.predict.confidence import build_track_record
from football_edge.trader.desk import Desk, family
from football_edge.trader.legs import Leg, LegFactory, count_test, goal_test
from football_edge.trader.record import build_goal_record
from football_edge.trader.slip import Book, History

CARD_LINES = (3.5, 4.5, 5.5, 6.5)
FIRST_LABEL_SEASON = 2019  # first season with calibrated out-of-sample predictions


@dataclass
class GridModel:
    """Enough of DixonColes for LegFactory: the grid of already-calibrated expected goals."""

    rho: float
    delta: float
    calibration: None = None

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        return adjust_draws(score_grid(lam, mu, self.rho), self.delta)


class _Replay(Desk):
    def __init__(self, conf: dict[str, Any], state: dict[str, Any]) -> None:
        self.conf, self._st = conf, state

    def _state(self) -> dict[str, Any]:
        return self._st


_G: dict[str, Any] = {}


def _won(leg: Leg, hg: int, ag: int, cards: float) -> bool | None:
    if leg.market == "cards":
        return (
            None
            if np.isnan(cards)
            else bool(count_test(leg.sel, float(leg.line or 0))(np.array([cards]))[0])
        )
    return bool(goal_test(leg.market, leg.sel, leg.line)(np.array([hg]), np.array([ag]))[0])


def _match(i: int) -> tuple[list[dict[str, Any]], Any]:
    p, conf = _G["preds"], _G["conf"]
    r = p.iloc[i]
    mid = f"m{i}"
    info = {
        "id": mid,
        "league": r.league,
        "kickoff_utc": str(r.date),
        "home": r.home,
        "away": r.away,
    }
    f = pd.Series(
        {
            "home": r.home,
            "away": r.away,
            "p_h": r.p_h,
            "p_d": r.p_d,
            "p_a": r.p_a,
            "p_over25": r.p_over25,
            "p_btts_yes": r.p_btts,
            "exp_home": r.exp_home,
            "exp_away": r.exp_away,
        }
    )
    counts: dict[tuple[str, float], float] = {}
    crow = _G["cards"].get(int(r.row))
    if crow is not None:
        for line in CARD_LINES:
            v = crow.get(f"p_team_{line}")
            if v is not None and not pd.isna(v):
                f[f"p_cards_over_{line}"] = float(v)
                counts[("cards", line)] = float(v)
    model = GridModel(_G["rho"], float(getattr(r, "cal_delta", 0.0) or 0.0))
    legs = _G["factories"][int(r.season)].for_match(info, f, model)
    grid = model.score_matrix(float(r.exp_home), float(r.exp_away))
    book = Book({mid: grid}, {mid: counts}, _G["hist"][pd.Timestamp(r.snapshot)])
    d = _Replay(conf, {"legs": {mid: legs}, "book": book})
    cards = float(_G["card_totals"][int(r.row)])
    hg, ag = int(r.goals_home), int(r.goals_away)
    base = {
        "i": i,
        "date": r.date,
        "snapshot": r.snapshot,
        "league": r.league,
        "season": int(r.season),
    }
    out: list[dict[str, Any]] = []
    singles: list[Leg] = []
    seen: set[str] = set()
    for x in d._candidates(mid):
        if family(x.market) not in seen:
            seen.add(family(x.market))
            singles.append(x)
    for k, x in enumerate(singles[: int(conf.get("singles_per_match", 4))]):
        out.append(
            {
                **base,
                "kind": "single",
                "rank": k,
                "labels": x.label,
                "markets": x.market,
                "p": x.p,
                "n_legs": 1,
                "won": _won(x, hg, ag, cards),
            }
        )
    pool = d._pool(mid, float(conf.get("parlay_leg_max", 0.95)))
    for kind, rule in d.kinds().items():
        combo = d.best_parlay(pool, rule["floor"], rule["max_legs"], rule["high_only"])
        if combo:
            ws = [_won(x, hg, ag, cards) for x in combo]
            out.append(
                {
                    **base,
                    "kind": kind,
                    "rank": 0,
                    "labels": " + ".join(x.label for x in combo),
                    "markets": "|".join(x.market for x in combo),
                    "p": book.chance(combo),
                    "n_legs": len(combo),
                    "won": None if any(w is None for w in ws) else all(ws),
                }
            )
    first = singles[0] if singles else None
    won_first = _won(first, hg, ag, cards) if first is not None else None
    return out, (i, first, won_first)


def _init(state: dict[str, Any]) -> None:
    _G.update(state)


def replay(
    preds: pd.DataFrame,
    counts: pd.DataFrame,
    matches: pd.DataFrame,
    conf: dict[str, Any],
    rho: float,
    seasons: tuple[int, int] = (2020, 2025),
    workers: int = 2,
) -> pd.DataFrame:
    """One row per slip the desk would have built, graded."""
    lo, hi = seasons
    p = preds[preds["season"].between(lo, hi)].reset_index(drop=True)
    factories = {}
    for s in range(lo, hi + 1):
        past = (FIRST_LABEL_SEASON, s - 1)
        rec = build_track_record(preds, counts, matches, seasons=past)
        extra = build_goal_record(preds, rho, seasons=past)
        record = pd.concat([rec, extra], ignore_index=True)
        factories[s] = LegFactory(
            record,
            tuple(conf.get("no_skill_markets", ["corners"])),
            {
                "cards": tuple(
                    float(x) for x in (conf.get("count_lines") or {}).get("cards", (3.5, 4.5, 5.5))
                )
            },
        )
    hist = {
        pd.Timestamp(w): History.from_matches(matches, before=pd.Timestamp(w))
        for w in p["snapshot"].unique()
    }
    c = counts[counts["stat"] == "cards"].set_index("row")
    card_rows = dict(zip((int(x) for x in c.index), c.to_dict("records"), strict=True))
    totals = (matches["HY"] + matches["AY"] + matches["HR"] + matches["AR"]).to_numpy(float)
    state = {
        "preds": p,
        "conf": conf,
        "rho": rho,
        "factories": factories,
        "hist": hist,
        "cards": card_rows,
        "card_totals": totals,
    }
    with Pool(workers, initializer=_init, initargs=(state,)) as pool:
        res = pool.map(_match, range(len(p)), chunksize=25)
    rows = [x for o, _ in res for x in o]
    # across matches: per day, one strong leg per match (the top of the board)
    _init(state)
    firsts = [f for _, f in res if f[1] is not None]
    day = pd.Series([p.iloc[i]["date"] for i, *_ in firsts])
    d = _Replay(conf, {"legs": {}, "book": Book({}, {}, None)})
    for dte, idx in day.groupby(day).groups.items():
        group = [firsts[k] for k in idx]
        legs = [g[1] for g in group]
        won = {g[1].key: g[2] for g in group}
        grids, cnts = {}, {}
        for g in group:
            r = p.iloc[g[0]]
            grids[g[1].match_id] = GridModel(
                rho, float(getattr(r, "cal_delta", 0.0) or 0.0)
            ).score_matrix(float(r.exp_home), float(r.exp_away))
            crow = card_rows.get(int(r.row), {})
            cnts[g[1].match_id] = {
                ("cards", ln): float(crow[f"p_team_{ln}"])
                for ln in CARD_LINES
                if f"p_team_{ln}" in crow and not pd.isna(crow[f"p_team_{ln}"])
            }
        snap = pd.Timestamp(p.iloc[group[0][0]]["snapshot"])
        d._st = {"legs": {}, "book": Book(grids, cnts, hist[snap])}
        legs.sort(key=lambda x: x.p, reverse=True)
        r0 = p.iloc[group[0][0]]
        for kind, rule in d.kinds().items():
            combo = d.best_parlay(legs, rule["floor"], rule["max_legs"], rule["high_only"])
            if combo:
                ws = [won[x.key] for x in combo]
                rows.append(
                    {
                        "i": -1,
                        "date": dte,
                        "snapshot": snap,
                        "league": "all",
                        "season": int(r0.season),
                        "kind": f"{kind}_across",
                        "rank": 0,
                        "labels": " + ".join(x.label for x in combo),
                        "markets": "|".join(x.market for x in combo),
                        "p": d._st["book"].chance(combo),
                        "n_legs": len(combo),
                        "won": None if any(w is None for w in ws) else all(ws),
                    }
                )
    return pd.DataFrame(rows)


NAMES = {
    "single": "Good singles",
    "single_top": "Top single per match",
    "safe": "Safe parlay (same match)",
    "risky": "Risky parlay (same match)",
    "safe_across": "Safe parlay across matches (per day)",
    "risky_across": "Risky parlay across matches (per day)",
}


def _streak(w: np.ndarray) -> int:
    best = cur = 0
    for x in w:
        cur = cur + 1 if not x else 0
        best = max(best, cur)
    return best


def summarise(slips: pd.DataFrame) -> pd.DataFrame:
    s = slips[slips["won"].notna()].copy()
    s["won"] = s["won"].astype(bool)
    s["block"] = (
        s["league"].astype(str) + "|" + pd.to_datetime(s["snapshot"]).dt.strftime("%Y-%m-%d")
    )
    groups = {
        k: s[s["kind"] == k] for k in ("single", "safe", "risky", "safe_across", "risky_across")
    }
    groups["single_top"] = s[(s["kind"] == "single") & (s["rank"] == 0)]
    rows = []
    for k in NAMES:
        g = groups[k].sort_values("date")
        if g.empty:
            continue
        diff = g["won"].to_numpy(float) - g["p"].to_numpy(float)
        lo, hi = block_bootstrap(diff, g["block"].to_numpy())
        rows.append(
            {
                "slip": NAMES[k],
                "n": len(g),
                "matches_or_days": int(g["block"].nunique()),
                "avg_legs": round(float(g["n_legs"].mean()), 2),
                "predicted": round(float(g["p"].mean()), 4),
                "landed": round(float(g["won"].mean()), 4),
                "gap": round(float(diff.mean()), 4),
                "gap_lo": round(lo, 4),
                "gap_hi": round(hi, 4),
                "longest_losing_run": _streak(g["won"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def per_season(slips: pd.DataFrame) -> pd.DataFrame:
    s = slips[slips["won"].notna()].copy()
    s["won"] = s["won"].astype(bool)
    t = (
        s.groupby(["kind", "season"])
        .agg(n=("p", "size"), predicted=("p", "mean"), landed=("won", "mean"))
        .round(4)
        .reset_index()
    )
    return t
