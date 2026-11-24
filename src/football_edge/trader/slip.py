"""The slip maths: the chance a whole slip wins, its risk, what each leg costs, the stake.

Joint chance
  * Different matches: independent, chances multiply.
  * Goal legs in the same match (result, double chance, goals, both score, exact score) are all
    questions about the final score, so their joint chance is read exactly from the model's
    score grid.
  * Corners and cards legs in the same match: each stat's legs combine into one range (over 8.5
    and under 11.5 = 9 to 11). Our count models are separate from the goals model, so how these
    move together with the goal legs comes from real matches: the lift
        P(all count ranges | the goal legs came true) / product of P(each count range)
    measured on league matches since 2020 (`lift_seasons`). With fewer than 300 such matches the
    lift is measured without the goal condition; it is kept between 0.5 and 1.5.

Risk grade (fixed rules, not a judgement call):
  * from the slip's chance: 60%+ Low, 40-60% Medium, 20-40% High, under 20% Very high;
  * one step riskier if any leg is weak (no model skill on that bet, or Low confidence).

Stake: a share of the monthly budget by risk grade (`trader.stake_by_risk`), rounded, never more
than what is left this month, and zero once the month's stop-loss is hit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_edge.trader.legs import GOAL_MARKETS, Leg, goal_test

GRADES = ("Low", "Medium", "High", "Very high")
MIN_LIFT_N = 300


def _grade(p: float) -> int:
    return 0 if p >= 0.60 else 1 if p >= 0.40 else 2 if p >= 0.20 else 3


@dataclass
class History:
    """Final scores and totals of past matches, for the count-leg lifts."""

    hg: np.ndarray
    ag: np.ndarray
    totals: dict[str, np.ndarray]

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, first_season: int = 2020,
                     before: pd.Timestamp | None = None) -> History:  # fmt: skip
        """`before`: only matches played before this date (replays must not see the future)."""
        m = matches[matches["season"] >= first_season]
        if before is not None:
            m = m[m["date"] < before]
        m = m.dropna(subset=["HC", "AC", "HY", "AY", "HR", "AR"])
        return cls(
            hg=m["FTHG"].to_numpy(int),
            ag=m["FTAG"].to_numpy(int),
            totals={
                "corners": (m["HC"] + m["AC"]).to_numpy(float),
                "cards": (m["HY"] + m["AY"] + m["HR"] + m["AR"]).to_numpy(float),
            },
        )


def _count_range(legs: list[Leg]) -> tuple[float, float]:
    """Over/under legs of one stat -> open range (lo, hi): total > lo and total < hi."""
    lo = max((x.line or 0.0 for x in legs if x.sel == "over"), default=-1.0)
    hi = min((x.line or 0.0 for x in legs if x.sel == "under"), default=float("inf"))
    return lo, hi


def match_chance(
    legs: list[Leg],
    grid: np.ndarray | None,
    hist: History | None,
    count_p: dict[tuple[str, float], float],
) -> float:
    """Joint chance of all legs of ONE match. `count_p[(stat, line)]` = model P(total > line)."""
    goal = [x for x in legs if x.market in GOAL_MARKETS]
    p_goal = 1.0
    if goal:
        if grid is None:  # no grid: fall back to the product (only in tests)
            p_goal = float(np.prod([x.p for x in goal]))
        else:
            idx = np.indices(grid.shape)
            h, a = idx[0], idx[1]
            ok = np.ones(grid.shape, dtype=bool)
            for x in goal:
                ok &= goal_test(x.market, x.sel, x.line)(h, a)
            p_goal = float(grid[ok].sum())
    by_stat: dict[str, list[Leg]] = defaultdict(list)
    for x in legs:
        if x.market in ("corners", "cards"):
            by_stat[x.market].append(x)
    p_counts, ranges = 1.0, {}
    for stat, xs in by_stat.items():
        lo, hi = _count_range(xs)
        if lo >= hi - 0.5:  # e.g. over 10.5 and under 9.5: cannot both happen
            return 0.0
        p_lo = count_p.get((stat, lo), 1.0) if lo >= 0 else 1.0
        p_hi = count_p.get((stat, hi), 0.0) if hi != float("inf") else 0.0
        p_counts *= max(p_lo - p_hi, 0.0)
        ranges[stat] = (lo, hi)
    lift = 1.0
    if ranges and hist is not None and len(hist.hg) and (goal or len(ranges) > 1):
        inside = {
            s: (hist.totals[s] > lo) & (hist.totals[s] < hi) for s, (lo, hi) in ranges.items()
        }
        both = np.logical_and.reduce(list(inside.values()))
        marg = float(np.prod([v.mean() for v in inside.values()]))
        cond = np.ones(len(hist.hg), dtype=bool)
        for x in goal:
            cond &= goal_test(x.market, x.sel, x.line)(hist.hg, hist.ag)
        use = cond if cond.sum() >= MIN_LIFT_N else np.ones_like(cond)
        if marg > 0 and use.sum():
            lift = float(np.clip(both[use].mean() / marg, 0.5, 1.5))
    joint = p_goal * p_counts * lift
    return float(min(joint, *(x.p for x in legs)))


@dataclass
class Book:
    """What the desk needs to price slips: per-match grids and count chances, the history."""

    grids: dict[str, np.ndarray]
    counts: dict[str, dict[tuple[str, float], float]]
    hist: History | None

    def chance(self, legs: list[Leg]) -> float:
        by_match: dict[str, list[Leg]] = defaultdict(list)
        for x in legs:
            by_match[x.match_id].append(x)
        return (
            float(
                np.prod(
                    [
                        match_chance(xs, self.grids.get(mid), self.hist, self.counts.get(mid, {}))
                        for mid, xs in by_match.items()
                    ]
                )
            )
            if legs
            else 0.0
        )


def check(
    legs: list[Leg],
    book: Book,
    staking: dict[str, Any],
    trader: dict[str, Any],
    month_profit: float = 0.0,
    pending: float = 0.0,
    price: float | None = None,
    exposure: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Everything the desk shows for one slip. `price`: the site's decimal odds for the slip.
    `exposure`: stake already riding on each match in open saved slips; together with this slip
    no match may carry more than `staking.max_match_fraction` of the monthly budget (a parlay's
    whole stake counts against every match in it)."""
    if not legs:
        raise ValueError("an empty slip")
    keys = [x.key for x in legs]
    if len(set(keys)) != len(keys):
        raise ValueError("the same leg twice")
    if len(legs) > 1 and any(x.market == "score" for x in legs):
        raise ValueError("a correct score is a single bet only, never part of a parlay")
    p = book.chance(legs)
    weak = [x for x in legs if x.weak]
    g = _grade(p) if p > 0 else 3
    if weak:
        g = min(g + 1, 3)
    risk = GRADES[g]
    costs = []
    if len(legs) > 1:
        for x in legs:
            without = book.chance([y for y in legs if y is not x])
            costs.append({"key": x.key, "label": x.label, "with": p, "without": without})
    budget = float(staking.get("monthly_budget", 0))
    frac = float((trader.get("stake_by_risk") or {}).get(risk, 0.0))
    step = float(staking.get("stake_rounding", 1)) or 1.0
    stop = month_profit <= -float(staking.get("stop_loss_fraction", 0.3)) * budget
    left = max(budget + month_profit - pending, 0.0)
    stake = 0.0 if stop else min(float(np.floor(budget * frac / step) * step), left)
    cap_note = ""
    cap = float(staking.get("max_match_fraction", 0.0)) * budget
    if cap > 0 and not stop:
        exposure = exposure or {}
        mids = sorted({x.match_id for x in legs})
        room = min(cap - exposure.get(m, 0.0) for m in mids)
        room = max(float(np.floor(room / step) * step), 0.0)
        if room < stake:
            stake = room
            busiest = max(mids, key=lambda m: exposure.get(m, 0.0))
            cap_note = (
                f"; cut to stay within {cap:,.0f} on one match "
                f"({exposure.get(busiest, 0.0):,.0f} already riding on it)"
            )
    kind = (
        "single"
        if len(legs) == 1
        else ("same-match" if len({x.match_id for x in legs}) == 1 else "parlay")
    )
    out: dict[str, Any] = {
        "kind": kind,
        "legs": [x.to_dict() for x in legs],
        "p": p,
        "fair_odds": (1 / p) if p > 0 else None,
        "risk": risk,
        "weak_legs": [x.label for x in weak],
        "leg_costs": costs,
        "stake": stake,
        "currency": str(staking.get("currency", "")),
        "stake_note": "stop-loss hit this month: no new stakes"
        if stop
        else f"{frac:.2%} of the monthly budget for a {risk} risk slip{cap_note}",
        "price": None,
    }
    if price is not None:
        if price <= 1:
            raise ValueError("the price must be decimal odds above 1, e.g. 3.40")
        back = p * price * 100
        out["price"] = {
            "odds": price,
            "back_per_100": back,
            "verdict": "above our fair price" if price * p >= 1 else "below our fair price",
            "note": "Our model's own view. Tests found the market usually more accurate, and a "
            "parlay pays the site's margin on every leg.",
        }
    return out
