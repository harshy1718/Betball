"""Bankroll simulation: what the staking policy would have done, week by week.

Start with a bankroll, then for every week of the test seasons:
  1. for each match and market, pick the most likely outcome (by model or by market);
  2. stake a tier share of the balance at the start of the week (ADR 0006 policy:
     confidence tiers, price guard, 4% per-match cap, up to 2 doubles, monthly stop-loss);
  3. settle at the prices actually on offer (Bet365 at kick-off, or a worse "8% site").

Probabilities come from the walk-forward run (trained only on earlier matches). Markets
are those with real prices in the data: 1X2, over/under 2.5, Asian handicap. BTTS, cards
and corners have no prices in the data, so they cannot be simulated honestly.

Asian handicap: line L is the home side's handicap (AHCh). With goal difference d
(home minus away), a home bet's margin is d + L, an away bet's -(d + L). Quarter lines
(e.g. -0.75) split the stake over the two neighbouring half lines (-0.5 and -1.0); each
half wins (> 0), pushes (= 0) or loses (< 0).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from football_edge.betting.staking import TIER_NAMES, StakingPolicy
from football_edge.models.dixon_coles import DixonColes

log = logging.getLogger(__name__)

GD_RANGE = range(-10, 11)  # goal differences tracked from the score grid


def gd_col(k: int) -> str:
    """Column name for P(goal difference = k); "m" for minus keeps names valid identifiers."""
    return f"gd_m{-k}" if k < 0 else f"gd_{k}"


class GridDixonColes(DixonColes):
    """Dixon-Coles that also reports P(goal difference = k), for Asian handicap."""

    def predict_extra(self, test: pd.DataFrame) -> dict[str, np.ndarray]:
        extra = super().predict_extra(test)
        lam, mu = self.expected_goals(test)
        g = np.arange(self.params.max_goals + 1)
        diff = g[:, None] - g[None, :]
        out = {gd_col(k): np.zeros(len(test)) for k in GD_RANGE}
        for i in range(len(test)):
            m = self.score_matrix(float(lam[i]), float(mu[i]))
            for k in GD_RANGE:
                out[gd_col(k)][i] = m[diff == k].sum()
        return {**extra, **out}


# ------------------------------------------------------------------ Asian handicap
def ah_halves(line: float) -> tuple[float, float]:
    """A quarter line splits into its two neighbouring half lines; others are doubled."""
    if round(line * 4) % 2:  # quarter line, e.g. -0.75 -> (-0.5, -1.0)
        return line + 0.25, line - 0.25
    return line, line


def ah_return(margin_halves: tuple[float, float], odds: float) -> float:
    """Profit per unit stake given each half's margin (goal diff plus handicap)."""
    r = 0.0
    for x in margin_halves:
        r += 0.5 * ((odds - 1) if x > 0 else (0.0 if x == 0 else -1.0))
    return r


def ah_win_loss(gd_probs: np.ndarray, line: float, side: str) -> tuple[float, float]:
    """(W, L): stake-weighted probability of winning / losing an Asian handicap bet."""
    ks = np.array(list(GD_RANGE), dtype=float)
    w = lo = 0.0
    for half in ah_halves(line):
        x = ks + half if side == "home" else -(ks + half)
        w += 0.5 * float(gd_probs[x > 0].sum())
        lo += 0.5 * float(gd_probs[x < 0].sum())
    return w, lo


# ------------------------------------------------------------------ prices
def read_book_prices(raw_dir: Path) -> pd.DataFrame:
    """Bet365 and Pinnacle prices at kick-off, plus the Asian handicap line, per match."""
    cols = {
        "B365CH": "b365_h",
        "B365CD": "b365_d",
        "B365CA": "b365_a",
        "B365C>2.5": "b365_over",
        "B365C<2.5": "b365_under",
        "AHCh": "ah_line",
        "B365CAHH": "b365_ah_home",
        "B365CAHA": "b365_ah_away",
        "PCAHH": "pin_ah_home",
        "PCAHA": "pin_ah_away",
    }
    parts = []
    for path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
        df = df[df["Date"].notna()]
        out = pd.DataFrame(
            {
                "league": path.name.split("_")[0],
                "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
                "home": df["HomeTeam"].astype(str).str.strip(),
                "away": df["AwayTeam"].astype(str).str.strip(),
            }
        )
        for src, dst in cols.items():
            v = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
            out[dst] = v
        parts.append(out)
    prices = pd.concat(parts, ignore_index=True)
    for c in cols.values():
        if c != "ah_line":
            prices[c] = prices[c].where(prices[c] > 1.0)
    return prices.drop_duplicates(["league", "date", "home", "away"])


def with_margin(odds: np.ndarray, margin: float) -> np.ndarray:
    """Re-price a market so its overround is `margin` (e.g. 0.08), keeping its shape."""
    q = 1 / odds
    q = q / q.sum(axis=1, keepdims=True) * (1 + margin)
    out: np.ndarray = 1 / q
    return out


# ------------------------------------------------------------------ candidates
def _choose_two_way(p: np.ndarray) -> int:
    return int(np.argmax(p))


def _fair(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    out: np.ndarray = inv / inv.sum()
    return out


def candidates(preds: pd.DataFrame, confidence: str, site_margin: float | None) -> pd.DataFrame:
    """One row per match and market: the chosen side, its probabilities, price and result.

    `confidence`: "model" (our walk-forward probabilities) or "market" (Pinnacle at
    kick-off, margin removed). Prices are Bet365 at kick-off, or re-priced to
    `site_margin` to imitate a softer site.
    """
    a = {c: preds[c].to_numpy() for c in preds.columns}
    gd_all = preds[[gd_col(k) for k in GD_RANGE]].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    def priced(*cols: str) -> np.ndarray | None:
        odds = np.array([float(a[c][i]) for c in cols])
        if np.isnan(odds).any():
            return None
        return with_margin(odds[None], site_margin)[0] if site_margin is not None else odds

    for i in range(len(preds)):
        d = int(a["goals_home"][i] - a["goals_away"][i])
        total = int(a["goals_home"][i] + a["goals_away"][i])
        base = {
            "week": a["snapshot"][i],
            "row": a["row"][i],
            "date": a["date"][i],
            "season": a["season"][i],
        }
        # 1X2
        odds = priced("b365_h", "b365_d", "b365_a")
        if odds is not None:
            model = np.array([a["p_h"][i], a["p_d"][i], a["p_a"][i]], dtype=float)
            mkt = np.array([a["close_h"][i], a["close_d"][i], a["close_a"][i]], dtype=float)
            use_mkt = confidence == "market" and not np.isnan(mkt).any()
            p = _fair(mkt) if use_mkt else model
            k = _choose_two_way(p)
            won = (d > 0, d == 0, d < 0)[k]
            rows.append(
                {
                    **base,
                    "market": "1x2",
                    "pick": ("home", "draw", "away")[k],
                    "w": p[k],
                    "l": 1 - p[k],
                    "odds": odds[k],
                    "ret": (odds[k] - 1) if won else -1.0,
                }
            )
        # Over/under 2.5
        odds = priced("b365_over", "b365_under")
        if odds is not None:
            over = float(a["p_over25"][i])
            model = np.array([over, 1 - over])
            mkt = np.array([a["ou_close_over"][i], a["ou_close_under"][i]], dtype=float)
            use_mkt = confidence == "market" and not np.isnan(mkt).any()
            p = _fair(mkt) if use_mkt else model
            k = _choose_two_way(p)
            won = (total >= 3, total < 3)[k]
            rows.append(
                {
                    **base,
                    "market": "ou25",
                    "pick": ("over", "under")[k],
                    "w": p[k],
                    "l": 1 - p[k],
                    "odds": odds[k],
                    "ret": (odds[k] - 1) if won else -1.0,
                }
            )
        # Asian handicap
        odds = priced("b365_ah_home", "b365_ah_away")
        line = float(a["ah_line"][i])
        if odds is not None and not np.isnan(line):
            wl = [ah_win_loss(gd_all[i], line, s) for s in ("home", "away")]
            pin = np.array([a["pin_ah_home"][i], a["pin_ah_away"][i]], dtype=float)
            if confidence == "market" and not np.isnan(pin).any():
                f = _fair(pin)
                wl = [(float(f[0]), float(f[1])), (float(f[1]), float(f[0]))]
            conf = np.array([w / (w + lo) if (w + lo) > 0 else 0.0 for w, lo in wl])
            k = _choose_two_way(conf)
            side = ("home", "away")[k]
            h1, h2 = ah_halves(line)
            margins = (d + h1, d + h2) if side == "home" else (-(d + h1), -(d + h2))
            shown = line if side == "home" else -line
            rows.append(
                {
                    **base,
                    "market": "ah",
                    "pick": f"{side} {shown:+g}",
                    "w": wl[k][0],
                    "l": wl[k][1],
                    "odds": odds[k],
                    "ret": ah_return(margins, float(odds[k])),
                }
            )
    out = pd.DataFrame(rows)
    out["conf"] = out["w"] / (out["w"] + out["l"])
    out["ev"] = out["w"] * (out["odds"] - 1) - out["l"]
    return out


# ------------------------------------------------------------------ the simulation
@dataclass
class SimResult:
    start: float
    final: float
    weekly: pd.DataFrame  # week, balance after settling
    bets: pd.DataFrame  # every bet placed
    stop_weeks: int
    lowest: float
    notes: list[str] = field(default_factory=list)


def _tier(policy: StakingPolicy, market: str, p: float) -> str | None:
    t = policy.tier_for("ou25" if market == "ah" else market, p)
    return t.name if t else None


def simulate(cands: pd.DataFrame, policy: StakingPolicy, start: float) -> SimResult:
    """Run the weekly loop. Stakes are tier shares of the balance at the start of the week."""
    fraction = {
        m: {t.name: t.fraction for t in policy.tiers["ou25" if m == "ah" else m]}
        for m in ("1x2", "ou25", "ah")
    }
    balance = start
    month: str | None = None
    month_start, stopped, stop_weeks = start, False, 0
    weekly, placed = [], []
    lowest = start
    double_tiers = TIER_NAMES[TIER_NAMES.index(policy.doubles_min_tier) :]
    for week, g in cands.sort_values(["week", "row"]).groupby("week", sort=True):
        m = str(week)[:7]  # year-month
        if m != month:
            month, month_start, stopped = m, balance, False
        if stopped or balance < policy.stake_rounding:
            stop_weeks += int(stopped)
            weekly.append((week, balance))
            continue
        g = g.copy()
        g["tier"] = [_tier(policy, mk, p) for mk, p in zip(g["market"], g["conf"], strict=True)]
        g["stake"] = [
            fraction[mk][t] * balance
            if isinstance(t, str) and ev >= -policy.price_tolerance
            else 0.0
            for mk, t, ev in zip(g["market"], g["tier"], g["ev"], strict=True)
        ]
        cap = policy.max_match_fraction * balance
        per_match = g.groupby("row")["stake"].transform("sum")
        g["stake"] *= np.where(per_match > cap, cap / per_match.where(per_match > 0, 1), 1.0)
        step = policy.stake_rounding
        g["stake"] = np.floor(g["stake"] / step) * step
        if g["stake"].sum() > balance:
            g["stake"] = np.floor(g["stake"] * balance / g["stake"].sum() / step) * step
        singles = g[g["stake"] > 0]

        # Doubles: two high-tier legs from different matches (1X2 and over/under only).
        legs = singles[singles["market"].isin(["1x2", "ou25"]) & singles["tier"].isin(double_tiers)]
        dbl = []
        for (_, a), (_, b) in itertools.combinations(legs.iterrows(), 2):
            if a["row"] == b["row"]:
                continue
            p, odds = a["conf"] * b["conf"], a["odds"] * b["odds"]
            if p * odds - 1 < -2 * policy.price_tolerance:
                continue
            won = a["ret"] > 0 and b["ret"] > 0
            dbl.append({"p": p, "odds": odds, "ret": (odds - 1) if won else -1.0})
        dbl = sorted(dbl, key=lambda x: -x["p"])[: policy.doubles_count]
        left = balance - singles["stake"].sum()
        d_stake = np.floor(min(policy.doubles_fraction * balance, max(left, 0) / 2) / step) * step

        profit = float((singles["stake"] * singles["ret"]).sum())
        placed += singles.assign(profit=singles["stake"] * singles["ret"], week=week)[
            [
                "week",
                "date",
                "season",
                "market",
                "pick",
                "tier",
                "conf",
                "odds",
                "ev",
                "stake",
                "profit",
            ]
        ].to_dict("records")
        for x in dbl:
            if d_stake > 0:
                profit += d_stake * x["ret"]
                placed.append(
                    {
                        "week": week,
                        "date": pd.NaT,
                        "season": int(g["season"].to_numpy()[0]),
                        "market": "double",
                        "pick": "2 legs",
                        "tier": "high",
                        "conf": x["p"],
                        "odds": x["odds"],
                        "ev": x["p"] * x["odds"] - 1,
                        "stake": d_stake,
                        "profit": d_stake * x["ret"],
                    }
                )
        balance += profit
        lowest = min(lowest, balance)
        if balance - month_start <= -policy.stop_loss_fraction * month_start:
            stopped = True
        weekly.append((week, balance))
    return SimResult(
        start=start,
        final=balance,
        weekly=pd.DataFrame(weekly, columns=["week", "balance"]),
        bets=pd.DataFrame(placed),
        stop_weeks=stop_weeks,
        lowest=lowest,
    )


def by_market(bets: pd.DataFrame) -> pd.DataFrame:
    """Bets, staked, profit, return on stake, and average confidence per market."""
    g = bets.groupby("market")
    out = pd.DataFrame(
        {
            "bets": g.size(),
            "staked": g["stake"].sum(),
            "profit": g["profit"].sum(),
            "avg_conf": g["conf"].mean(),
            "win_rate": g["profit"].apply(lambda s: float((s > 0).mean())),
        }
    )
    out["return"] = out["profit"] / out["staked"]
    return out
