"""Confidence-based staking with guard rails (docs/adr/0006-staking-policy.md).

What this does, per fixture and market:
  1. The probability used is the market's fair probability (margin removed, sharp prices
     preferred) when prices are given, else the model's. The market's is more accurate
     (ADR 0007). Both are shown, and big disagreements are flagged.
  2. The selection is the most likely outcome; its probability sets a confidence tier,
     and the tier sets the stake (a fixed share of the monthly budget, not of whatever
     is left - so losing never makes stakes chase).
  3. A price guard: if the site's price is so short that the expected loss is worse than
     a normal bookmaker margin, the bet is skipped. `min_odds` tells you the lowest
     price worth taking even when you have not typed the site's price in.
  4. Caps: per match, per month (budget), and a stop-loss for the month.

What this does NOT do: create an edge. Our tests found none against the market. This
module manages risk and avoids the worst prices; on average it still pays the margin.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_edge.predict.forecast import MARKETS, PRICE_OF

TIER_NAMES = ("low", "normal", "high")


@dataclass(frozen=True)
class Tier:
    name: str
    min_prob: float
    fraction: float  # share of the monthly budget staked at this tier


@dataclass(frozen=True)
class StakingPolicy:
    monthly_budget: float
    currency: str
    tiers: dict[str, tuple[Tier, ...]]  # per market, ascending min_prob
    price_tolerance: float
    max_match_fraction: float
    stop_loss_fraction: float
    stake_rounding: float
    confidence: str  # "market" (fair market probability when available) or "model"
    disagree_at: float  # flag when model and market differ by at least this much
    doubles_count: int
    doubles_fraction: float
    doubles_min_tier: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StakingPolicy:
        tiers: dict[str, tuple[Tier, ...]] = {}
        for market, rows in d["tiers"].items():
            if market not in MARKETS:
                raise ValueError(f"unknown market in staking tiers: {market}")
            if len(rows) != len(TIER_NAMES):
                raise ValueError(f"{market}: give exactly {len(TIER_NAMES)} tiers")
            ts = tuple(
                Tier(n, float(p), float(f)) for n, (p, f) in zip(TIER_NAMES, rows, strict=True)
            )
            if any(a.min_prob >= b.min_prob for a, b in itertools.pairwise(ts)):
                raise ValueError(f"{market}: tier probabilities must increase")
            tiers[market] = ts
        dbl = d.get("doubles", {})
        policy = cls(
            monthly_budget=float(d["monthly_budget"]),
            currency=str(d.get("currency", "")),
            tiers=tiers,
            price_tolerance=float(d["price_tolerance"]),
            max_match_fraction=float(d["max_match_fraction"]),
            stop_loss_fraction=float(d["stop_loss_fraction"]),
            stake_rounding=float(d.get("stake_rounding", 1)),
            confidence=str(d.get("confidence", "market")),
            disagree_at=float(d.get("disagree_at", 0.08)),
            doubles_count=int(dbl.get("max_count", 0)),
            doubles_fraction=float(dbl.get("stake_fraction", 0.0)),
            doubles_min_tier=str(dbl.get("min_leg_tier", "high")),
        )
        if policy.monthly_budget <= 0:
            raise ValueError("monthly_budget must be positive")
        if policy.confidence not in ("market", "model"):
            raise ValueError("confidence must be 'market' or 'model'")
        if policy.doubles_min_tier not in TIER_NAMES:
            raise ValueError(f"doubles min_leg_tier must be one of {TIER_NAMES}")
        return policy

    def tier_for(self, market: str, p: float) -> Tier | None:
        found = None
        for t in self.tiers[market]:
            if p >= t.min_prob:
                found = t
        return found


@dataclass(frozen=True)
class MonthStatus:
    """Where the month stands, from the bet log."""

    settled_profit: float  # sum of profit on settled bets this month
    pending_stake: float  # money already on bets not yet settled

    def remaining(self, budget: float) -> float:
        return budget + self.settled_profit - self.pending_stake


def _round(x: float, step: float) -> float:
    return float(np.floor(x / step) * step) if step > 0 else x


def select_bets(
    fc: pd.DataFrame, policy: StakingPolicy, month: MonthStatus
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (singles, doubles, notes). Every candidate is listed, staked or not, with why."""
    notes: list[str] = []
    budget = policy.monthly_budget
    stop = month.settled_profit <= -policy.stop_loss_fraction * budget
    if stop:
        notes.append(
            f"STOP-LOSS: this month is {month.settled_profit:,.0f} {policy.currency}, past the "
            f"{policy.stop_loss_fraction:.0%} limit. No stakes until next month."
        )
    rows = []
    for k, f in fc.iterrows():
        for market, (names, pcols) in MARKETS.items():
            model_p = np.array([f[c] for c in pcols], dtype=float)
            mkt_p = np.array([f.get("mkt_" + c[2:], np.nan) for c in pcols], dtype=float)
            use_market = policy.confidence == "market" and not np.isnan(mkt_p).any()
            probs = mkt_p if use_market else model_p
            i = int(np.argmax(probs))
            p, pcol = float(probs[i]), pcols[i]
            gap = float(model_p[i] - mkt_p[i]) if not np.isnan(mkt_p).any() else np.nan
            price = f.get(PRICE_OF[pcol], np.nan)
            price = float(price) if pd.notna(price) else np.nan
            tier = policy.tier_for(market, p)
            ev = p * price - 1 if not np.isnan(price) else np.nan
            stake = 0.0
            if tier is None:
                status = "skip: probability too low"
            elif not np.isnan(ev) and ev < -policy.price_tolerance:
                status = "skip: price too short"
            elif stop:
                status = "skip: stop-loss"
            else:
                stake = tier.fraction * budget
                status = "bet" if not np.isnan(price) else "bet if price >= min_odds"
            rows.append(
                {
                    "fixture": k,
                    "league": f["league"],
                    "date": f["date"],
                    "home": f["home"],
                    "away": f["away"],
                    "market": market,
                    "selection": names[i],
                    "prob": p,
                    "prob_source": (
                        f"market ({f.get(f'mkt_src_{market}', '-')})" if use_market else "model"
                    ),
                    "model_prob": float(model_p[i]),
                    "market_prob": float(mkt_p[i]),
                    "disagree": bool(not np.isnan(gap) and abs(gap) >= policy.disagree_at),
                    "fair_odds": 1 / p,
                    "min_odds": (1 - policy.price_tolerance) / p,
                    "odds": price,
                    "exp_value": ev,
                    "tier": tier.name if tier else "-",
                    "stake": stake,
                    "status": status,
                }
            )
    singles = pd.DataFrame(rows)

    # Per-match cap: markets on the same match move together, so limit total exposure.
    cap = policy.max_match_fraction * budget
    per_match = singles.groupby("fixture")["stake"].transform("sum")
    scale = np.where(per_match > cap, cap / per_match.where(per_match > 0, 1), 1.0)
    singles["stake"] = singles["stake"] * scale

    # Monthly budget: never stake money that is not left this month.
    left = max(month.remaining(budget), 0.0)
    total = float(singles["stake"].sum())
    if total > left:
        notes.append(f"stakes scaled down to fit the {left:,.0f} {policy.currency} left this month")
        singles["stake"] *= left / total if total else 0.0
    singles["stake"] = [_round(s, policy.stake_rounding) for s in singles["stake"]]
    singles.loc[(singles["stake"] == 0) & (singles["status"].str.startswith("bet")), "status"] = (
        "skip: stake rounds to 0"
    )

    doubles = _doubles(singles, policy, left - float(singles["stake"].sum()), stop)
    return singles, doubles, notes


def _doubles(singles: pd.DataFrame, policy: StakingPolicy, left: float, stop: bool) -> pd.DataFrame:
    """At most two legs, from different matches, each leg priced and confident enough."""
    cols = ["leg1", "leg2", "prob", "odds", "exp_value", "stake"]
    if stop or policy.doubles_count <= 0 or left <= 0:
        return pd.DataFrame(columns=cols)
    min_rank = TIER_NAMES.index(policy.doubles_min_tier)
    ok = singles[
        (singles["stake"] > 0)
        & singles["odds"].notna()
        & singles["tier"].map(lambda t: t in TIER_NAMES and TIER_NAMES.index(t) >= min_rank)
    ]
    cands = []
    for (_, a), (_, b) in itertools.combinations(ok.iterrows(), 2):
        if a["fixture"] == b["fixture"]:
            continue  # same match: legs are correlated, the combined price misleads
        p, odds = a["prob"] * b["prob"], a["odds"] * b["odds"]
        if p * odds - 1 < -2 * policy.price_tolerance:
            continue
        cands.append(
            {
                "leg1": leg_key(a),
                "leg2": leg_key(b),
                "prob": p,
                "odds": odds,
                "exp_value": p * odds - 1,
                "stake": _round(
                    min(
                        policy.doubles_fraction * policy.monthly_budget, left / policy.doubles_count
                    ),
                    policy.stake_rounding,
                ),
            }
        )
    out = pd.DataFrame(cands, columns=cols)
    return (
        out.sort_values("prob", ascending=False).head(policy.doubles_count).reset_index(drop=True)
    )


def leg_key(r: pd.Series) -> str:
    """The text that identifies one selection: league|date|home|away|market|selection."""
    return "|".join(
        [r["league"], f"{r['date']:%Y-%m-%d}", r["home"], r["away"], r["market"], r["selection"]]
    )
