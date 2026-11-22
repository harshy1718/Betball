"""Legs: every bet the trader desk can use, for one match, with its chance and trust.

A leg is one selection in one match: a result, double chance, goals over/under 2.5, both teams
to score, an exact score, total corners or total cards over/under a line. Each leg carries:

  p           our model's chance (never the market's: staking.confidence is "model")
  confidence  High / Medium / Low from the leg's own track record at that probability
  skill       does the model beat a plain base rate on this kind of bet at all?

Corners have no real skill (ADR 0008): the model is no better than the league average. Such
legs are allowed but always flagged as weak.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_edge.predict.confidence import lookup
from football_edge.trader.record import goal_markets

# Lines Harsh's site offers (22 Nov 2026); configurable as trader.count_lines.
COUNT_LINES = {"corners": (8.5, 9.5, 10.5), "cards": (3.5, 4.5, 5.5)}
EXTRA = ("goals", "home_goals", "away_goals", "hcp", "wtn", "oddeven")
GOAL_MARKETS = ("1x2", "dc", "ou25", "btts", "score", *EXTRA)
_EXTRA_TESTS = {(m, sel): t for m, sels in goal_markets().items() for sel, t in sels}


def record_key(market: str, sel: str, line: float | None) -> str:
    """The track-record market for an extra goal leg ('home-1.5' -> 'hcp_home2')."""
    if market in ("goals", "home_goals", "away_goals"):
        return f"{market}_{line}"
    if market == "hcp":
        side, k = sel[:4], int(float(sel[5:]) + 0.5)  # "home-1.5" -> ("home", 2)
        fav = side if sel[4] == "-" else ("away" if side == "home" else "home")
        return f"hcp_{fav}{k}"
    return market  # wtn, oddeven


@dataclass(frozen=True)
class Leg:
    match_id: str
    league: str
    kickoff_utc: str
    home: str  # display names
    away: str
    home_key: str  # our data's names
    away_key: str
    market: str  # 1x2 | dc | ou25 | btts | score | corners | cards
    sel: str  # home/draw/away | 1X/X2/12 | over/under | yes/no | "2-1"
    line: float | None
    label: str
    p: float
    confidence: str
    skill: bool

    @property
    def key(self) -> str:
        return f"{self.match_id}|{self.market}|{self.sel}|{self.line if self.line else ''}"

    @property
    def weak(self) -> bool:
        return (not self.skill) or self.confidence in ("Low", "-")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key, "weak": self.weak}


# -- what each leg means, as a test on the final score / totals --------------------------
def goal_test(
    market: str, sel: str, line: float | None = None
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """For goal legs: a function of (home goals, away goals) arrays -> won?"""
    if market in EXTRA:
        return _EXTRA_TESTS[(record_key(market, sel, line), sel)]
    if market == "1x2":
        return {
            "home": lambda h, a: h > a,
            "draw": lambda h, a: h == a,
            "away": lambda h, a: h < a,
        }[sel]
    if market == "dc":
        return {"1X": lambda h, a: h >= a, "X2": lambda h, a: h <= a, "12": lambda h, a: h != a}[
            sel
        ]
    if market == "ou25":
        return (lambda h, a: h + a >= 3) if sel == "over" else (lambda h, a: h + a < 3)
    if market == "btts":
        if sel == "yes":
            return lambda h, a: (h > 0) & (a > 0)
        return lambda h, a: (h == 0) | (a == 0)
    if market == "score":
        i, j = (int(x) for x in sel.split("-"))
        return lambda h, a: (h == i) & (a == j)
    raise ValueError(f"not a goal market: {market}")


def count_test(sel: str, line: float) -> Callable[[np.ndarray], np.ndarray]:
    return (lambda t: t > line) if sel == "over" else (lambda t: t < line)


# -- building legs from one forecast row ---------------------------------------------------
class LegFactory:
    def __init__(
        self,
        record: pd.DataFrame | None,
        no_skill: tuple[str, ...] = ("corners",),
        count_lines: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        """`no_skill`: bet types treated as having no model skill whatever the track record
        says. Corners: the team model is no better than the league average (ADR 0008).
        `count_lines`: the corners and cards lines offered (default COUNT_LINES)."""
        self.record = record
        self.no_skill = no_skill
        self.count_lines = count_lines or COUNT_LINES
        self._skill: dict[str, bool] = {}
        if record is not None:
            mod = record[record["source"] == "model"]
            self._skill = {str(m): bool(g["skill"].iloc[0]) for m, g in mod.groupby("market")}

    def trust(self, market: str, p: float) -> tuple[str, bool]:
        """(confidence label, skill) for a probability on one track-record market."""
        if self.record is None:
            return "-", False
        label = lookup(self.record, "model", market, p)[0]
        if market.split("_")[0] in self.no_skill:
            return label, False
        return label, self._skill.get(market, False)

    @staticmethod
    def _extra_bets(hn: str, an: str) -> list[tuple[str, str, float | None, str]]:
        """(market, selection, line, label) for every extra goal bet."""
        out: list[tuple[str, str, float | None, str]] = []
        for line in (0.5, 1.5, 3.5, 4.5):
            out += [
                ("goals", "over", line, f"Over {line:g} goals"),
                ("goals", "under", line, f"Under {line:g} goals"),
            ]
        for market, team in (("home_goals", hn), ("away_goals", an)):
            for line in (0.5, 1.5, 2.5):
                out += [
                    (market, "over", line, f"{team} over {line:g} goals"),
                    (market, "under", line, f"{team} under {line:g} goals"),
                ]
        for k in (1.5, 2.5):
            out += [
                ("hcp", f"home-{k:g}", None, f"{hn} -{k:g} (win by {int(k + 0.5)}+)"),
                ("hcp", f"away+{k:g}", None, f"{an} +{k:g}"),
                ("hcp", f"away-{k:g}", None, f"{an} -{k:g} (win by {int(k + 0.5)}+)"),
                ("hcp", f"home+{k:g}", None, f"{hn} +{k:g}"),
            ]
        out += [
            ("wtn", "home", None, f"{hn} to win to nil"),
            ("wtn", "away", None, f"{an} to win to nil"),
            ("oddeven", "odd", None, "Odd number of goals"),
            ("oddeven", "even", None, "Even number of goals"),
        ]
        return out

    def for_match(self, info: dict[str, Any], f: pd.Series, model: Any) -> list[Leg]:
        """All legs for one match. `info`: the site's match summary (id, names, kickoff)."""
        base = {
            "match_id": info["id"],
            "league": info["league"],
            "kickoff_utc": info["kickoff_utc"],
            "home": info["home"],
            "away": info["away"],
            "home_key": str(f["home"]),
            "away_key": str(f["away"]),
        }
        hn, an = info["home"], info["away"]
        ph, pd_, pa = float(f["p_h"]), float(f["p_d"]), float(f["p_a"])
        legs: list[Leg] = []

        def add(
            market: str,
            sel: str,
            line: float | None,
            label: str,
            p: float,
            track: str,
            track_p: float | None = None,
        ) -> None:
            conf, skill = self.trust(track, p if track_p is None else track_p)
            legs.append(
                Leg(
                    **base,
                    market=market,
                    sel=sel,
                    line=line,
                    label=label,
                    p=float(p),
                    confidence=conf,
                    skill=skill,
                )
            )

        add("1x2", "home", None, f"{hn} win", ph, "1x2")
        add("1x2", "draw", None, "Draw", pd_, "1x2")
        add("1x2", "away", None, f"{an} win", pa, "1x2")
        # Double chance = "not the other outcome": as trustworthy as that outcome's chance.
        add("dc", "1X", None, f"{hn} or draw", ph + pd_, "1x2", pa)
        add("dc", "X2", None, f"{an} or draw", pa + pd_, "1x2", ph)
        add("dc", "12", None, f"{hn} or {an} (no draw)", ph + pa, "1x2", pd_)
        po = float(f["p_over25"])
        add("ou25", "over", 2.5, "Over 2.5 goals", po, "ou25")
        add("ou25", "under", 2.5, "Under 2.5 goals", 1 - po, "ou25")
        pb = float(f["p_btts_yes"])
        add("btts", "yes", None, "Both teams score", pb, "btts")
        add("btts", "no", None, "Not both teams score", 1 - pb, "btts")
        if model is not None:
            m = model.score_matrix(float(f["exp_home"]), float(f["exp_away"]))
            gh, ga = np.indices(m.shape)
            for market, sel, line, label in self._extra_bets(hn, an):
                p = float(m[goal_test(market, sel, line)(gh, ga)].sum())
                add(market, sel, line, label, p, record_key(market, sel, line))
            order = np.argsort(m, axis=None)[::-1][:3]
            for i, j in zip(*np.unravel_index(order, m.shape), strict=True):
                legs.append(
                    Leg(
                        **base,
                        market="score",
                        sel=f"{i}-{j}",
                        line=None,
                        label=f"Correct score {i}-{j}",
                        p=float(m[i, j]),
                        confidence="Low",
                        skill=False,
                    )
                )
        for stat, lines in self.count_lines.items():
            for line in lines:
                col = f"p_{stat}_over_{line}"
                if col not in f.index or pd.isna(f[col]):
                    continue
                p = float(f[col])
                add(stat, "over", line, f"Over {line:g} {stat}", p, f"{stat}_{line}")
                add(stat, "under", line, f"Under {line:g} {stat}", 1 - p, f"{stat}_{line}")
        return legs


# -- reading a bet typed in words (for the trader chat) -----------------------------------
_COUNT = re.compile(r"(over|under)\s*(\d+(?:\.5)?)\s*(corners?|cards?)")


def find_leg(legs: list[Leg], bet: str) -> Leg | None:
    """'home win', 'Liverpool win', 'draw', '1X', 'over 2.5', 'btts no', '2-1',
    'over 9.5 corners', 'cards under 4.5' -> the matching leg of one match."""
    b = " ".join(bet.lower().replace("goals", "").split())
    if not legs:
        return None
    hn, an = legs[0].home.lower(), legs[0].away.lower()
    m = _COUNT.search(b) or re.search(r"(corners?|cards?)\s*(over|under)\s*(\d+(?:\.5)?)", b)
    if m:
        g = m.groups()
        sel, line, stat = (g[0], g[1], g[2]) if g[0] in ("over", "under") else (g[1], g[2], g[0])
        stat = "corners" if stat.startswith("corner") else "cards"
        return next(
            (x for x in legs if x.market == stat and x.sel == sel and x.line == float(line)), None
        )
    score = re.fullmatch(r"(?:correct score\s*)?(\d+)\s*-\s*(\d+)", b)
    if score:
        sel = f"{score.group(1)}-{score.group(2)}"
        return next((x for x in legs if x.market == "score" and x.sel == sel), None)
    alias = {
        "home win": ("1x2", "home"),
        "home": ("1x2", "home"),
        f"{hn} win": ("1x2", "home"),
        f"{hn}": ("1x2", "home"),
        "away win": ("1x2", "away"),
        "away": ("1x2", "away"),
        f"{an} win": ("1x2", "away"),
        f"{an}": ("1x2", "away"),
        "draw": ("1x2", "draw"),
        "1x": ("dc", "1X"),
        "x2": ("dc", "X2"),
        "12": ("dc", "12"),
        "home or draw": ("dc", "1X"),
        "away or draw": ("dc", "X2"),
        "no draw": ("dc", "12"),
        f"{hn} or draw": ("dc", "1X"),
        f"{an} or draw": ("dc", "X2"),
        "over 2.5": ("ou25", "over"),
        "under 2.5": ("ou25", "under"),
        "over": ("ou25", "over"),
        "under": ("ou25", "under"),
        "btts": ("btts", "yes"),
        "btts yes": ("btts", "yes"),
        "btts no": ("btts", "no"),
        "both teams to score": ("btts", "yes"),
        "both teams score": ("btts", "yes"),
    }
    hit = alias.get(b.strip())
    if hit:
        return next((x for x in legs if (x.market, x.sel) == hit), None)
    return None
