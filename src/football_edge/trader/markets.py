"""More markets for the desk (ADR 0028): extra full-time bets read from the score grid, bets
with a refund (draw no bet, whole-line Asian handicaps and totals) and half-time bets from the
half model.

Every market is a test on the result (home goals, away goals, home goals at half-time, away goals
at half-time), so the same definition prices it from the model, grades it after the match and
builds its track record. Refund markets also have a `void` test: when it is true the stake comes
back and the bet is not graded; their chance is the chance of winning when there is no refund.

Half model (ADR 0028): each side's first-half goals are a binomial share of its full-time goals,
with the league's share of home / away goals scored before half-time:
    P(HT h1-a1, FT h-a) = grid(h, a) * Bin(h1; h, s_home) * Bin(a1; a, s_away)
so every full-time chance stays exactly the live model's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import binom

Arr = npt.NDArray[np.int64]
Test4 = Callable[..., Any]  # (home, away, half-time home, half-time away) -> bool array


@dataclass(frozen=True)
class Market:
    market: str  # leg market name
    sel: str
    line: float | None
    key: str  # track-record key
    label: str  # with {h} / {a} for the team names
    win: Test4
    void: Test4 | None = None
    half: bool = False
    family: str = ""


def _total_tests(line: float) -> tuple[Test4, Test4]:
    return (lambda h, a, h1, a1: h + a > line), (lambda h, a, h1, a1: h + a < line)


def catalogue() -> list[Market]:
    """Every market ADR 0028 adds, in display order."""
    out: list[Market] = []
    add = out.append
    # -- full time, win or lose ------------------------------------------------------------
    o, u = _total_tests(5.5)
    add(Market("goals", "over", 5.5, "goals_5.5", "Over 5.5 goals", o, family="total"))
    add(Market("goals", "under", 5.5, "goals_5.5", "Under 5.5 goals", u, family="total"))
    for side, home in (("home", True), ("away", False)):
        t = "{h}" if home else "{a}"
        add(
            Market(
                f"{side}_goals",
                "over",
                3.5,
                f"{side}_goals_3.5",
                f"{t} over 3.5 goals",
                (lambda h, a, h1, a1, hm=home: (h if hm else a) > 3.5),
                family=f"{side}_goals",
            )
        )
        add(
            Market(
                f"{side}_goals",
                "under",
                3.5,
                f"{side}_goals_3.5",
                f"{t} under 3.5 goals",
                (lambda h, a, h1, a1, hm=home: (h if hm else a) < 3.5),
                family=f"{side}_goals",
            )
        )
    for k in range(6):
        add(
            Market(
                "exact",
                str(k),
                None,
                f"exact_{k}",
                f"Exactly {k} goal{'' if k == 1 else 's'}",
                (lambda h, a, h1, a1, k=k: h + a == k),
                family="total",
            )
        )
    add(
        Market(
            "exact",
            "6+",
            None,
            "exact_6+",
            "6 or more goals",
            (lambda h, a, h1, a1: h + a >= 6),
            family="total",
        )
    )
    for lo, hi in ((0, 1), (2, 3), (4, 5)):
        add(
            Market(
                "range",
                f"{lo}-{hi}",
                None,
                f"range_{lo}-{hi}",
                f"{lo}-{hi} goals",
                (lambda h, a, h1, a1, lo=lo, hi=hi: (h + a >= lo) & (h + a <= hi)),
                family="total",
            )
        )
    for side, sign in (("home", 1), ("away", -1)):
        t = "{h}" if side == "home" else "{a}"
        for k in (1, 2):
            add(
                Market(
                    "margin",
                    f"{side}{k}",
                    None,
                    f"margin_{side}{k}",
                    f"{t} win by exactly {k}",
                    (lambda h, a, h1, a1, s=sign, k=k: s * (h - a) == k),
                    family="result",
                )
            )
        add(
            Market(
                "margin",
                f"{side}3+",
                None,
                f"margin_{side}3+",
                f"{t} win by 3 or more",
                (lambda h, a, h1, a1, s=sign: s * (h - a) >= 3),
                family="result",
            )
        )
    for side, home in (("home", True), ("away", False)):
        t = "{h}" if home else "{a}"
        for k in range(3):
            add(
                Market(
                    f"{side}_exact",
                    str(k),
                    None,
                    f"{side}_exact_{k}",
                    f"{t} score exactly {k}",
                    (lambda h, a, h1, a1, hm=home, k=k: (h if hm else a) == k),
                    family=f"{side}_goals",
                )
            )
        add(
            Market(
                f"{side}_exact",
                "3+",
                None,
                f"{side}_exact_3+",
                f"{t} score 3 or more",
                (lambda h, a, h1, a1, hm=home: (h if hm else a) >= 3),
                family=f"{side}_goals",
            )
        )
        add(
            Market(
                f"cs_{side}",
                "yes",
                None,
                f"cs_{side}",
                f"{t} keep a clean sheet",
                (lambda h, a, h1, a1, hm=home: (a if hm else h) == 0),
                family="clean_sheet",
            )
        )
        add(
            Market(
                f"cs_{side}",
                "no",
                None,
                f"cs_{side}",
                f"{t} concede",
                (lambda h, a, h1, a1, hm=home: (a if hm else h) > 0),
                family="clean_sheet",
            )
        )
    res: dict[str, Callable[[Arr, Arr], npt.NDArray[np.bool_]]] = {
        "home": lambda h, a: h > a,
        "draw": lambda h, a: h == a,
        "away": lambda h, a: h < a,
    }
    res_label = {"home": "{h} win", "draw": "Draw", "away": "{a} win"}
    for r, rt in res.items():
        for b, bt in (("yes", True), ("no", False)):
            add(
                Market(
                    "res_btts",
                    f"{r}-{b}",
                    None,
                    f"res_btts_{r}-{b}",
                    f"{res_label[r]} & {'both score' if bt else 'not both score'}",
                    (lambda h, a, h1, a1, rt=rt, bt=bt: rt(h, a) & (((h > 0) & (a > 0)) == bt)),
                    family="combo",
                )
            )
        for ou, over in (("over", True), ("under", False)):
            add(
                Market(
                    "res_ou",
                    f"{r}-{ou}",
                    None,
                    f"res_ou_{r}-{ou}",
                    f"{res_label[r]} & {ou} 2.5 goals",
                    (lambda h, a, h1, a1, rt=rt, over=over: rt(h, a) & ((h + a > 2.5) == over)),
                    family="combo",
                )
            )
    for ou, over in (("over", True), ("under", False)):
        for b, bt in (("yes", True), ("no", False)):
            add(
                Market(
                    "ou_btts",
                    f"{ou}-{b}",
                    None,
                    f"ou_btts_{ou}-{b}",
                    f"{ou.capitalize()} 2.5 & {'both score' if bt else 'not both score'}",
                    (
                        lambda h, a, h1, a1, over=over, bt=bt: (
                            ((h + a > 2.5) == over) & (((h > 0) & (a > 0)) == bt)
                        )
                    ),
                    family="combo",
                )
            )
    # -- full time with a refund (singles only) ---------------------------------------------
    for side, sign in (("home", 1), ("away", -1)):
        t = "{h}" if side == "home" else "{a}"
        add(
            Market(
                "dnb",
                side,
                None,
                f"dnb_{side}",
                f"{t} draw no bet",
                (lambda h, a, h1, a1, s=sign: s * (h - a) > 0),
                (lambda h, a, h1, a1: h == a),
                family="result",
            )
        )
        for k in (1, 2, 3):
            # "home -k": home wins by more than k, refund if exactly k; "home +k": home does
            # not lose by more than k, refund if it loses by exactly k.
            add(
                Market(
                    "ahc",
                    f"{side}-{k}",
                    None,
                    f"ahc_{side}-{k}",
                    f"{t} -{k} (Asian)",
                    (lambda h, a, h1, a1, s=sign, k=k: s * (h - a) > k),
                    (lambda h, a, h1, a1, s=sign, k=k: s * (h - a) == k),
                    family="result",
                )
            )
            add(
                Market(
                    "ahc",
                    f"{side}+{k}",
                    None,
                    f"ahc_{side}+{k}",
                    f"{t} +{k} (Asian)",
                    (lambda h, a, h1, a1, s=sign, k=k: s * (h - a) > -k),
                    (lambda h, a, h1, a1, s=sign, k=k: s * (h - a) == -k),
                    family="result",
                )
            )
    for line in (2.0, 3.0, 4.0):
        n = int(line)
        add(
            Market(
                "atot",
                "over",
                line,
                f"atot_{n}",
                f"Over {n} goals (Asian)",
                (lambda h, a, h1, a1, n=n: h + a > n),
                (lambda h, a, h1, a1, n=n: h + a == n),
                family="total",
            )
        )
        add(
            Market(
                "atot",
                "under",
                line,
                f"atot_{n}",
                f"Under {n} goals (Asian)",
                (lambda h, a, h1, a1, n=n: h + a < n),
                (lambda h, a, h1, a1, n=n: h + a == n),
                family="total",
            )
        )
    # -- halves -----------------------------------------------------------------------------
    for half, name in (("fh", "1st half"), ("sh", "2nd half")):

        def hg(h: Arr, h1: Arr, half: str = half) -> Arr:
            return h1 if half == "fh" else h - h1

        for r, lab in (("home", "{h} win"), ("draw", "Draw"), ("away", "{a} win")):
            add(
                Market(
                    f"{half}1x2",
                    r,
                    None,
                    f"{half}1x2_{r}",
                    f"{name}: {lab}",
                    (lambda h, a, h1, a1, r=r, hg=hg: res[r](hg(h, h1), hg(a, a1))),
                    half=True,
                    family=half,
                )
            )
        for line in (0.5, 1.5, 2.5):
            add(
                Market(
                    f"{half}goals",
                    "over",
                    line,
                    f"{half}goals_{line}",
                    f"{name}: over {line:g} goals",
                    (lambda h, a, h1, a1, line=line, hg=hg: hg(h, h1) + hg(a, a1) > line),
                    half=True,
                    family=half,
                )
            )
            add(
                Market(
                    f"{half}goals",
                    "under",
                    line,
                    f"{half}goals_{line}",
                    f"{name}: under {line:g} goals",
                    (lambda h, a, h1, a1, line=line, hg=hg: hg(h, h1) + hg(a, a1) < line),
                    half=True,
                    family=half,
                )
            )
        add(
            Market(
                f"{half}btts",
                "yes",
                None,
                f"{half}btts",
                f"{name}: both score",
                (lambda h, a, h1, a1, hg=hg: (hg(h, h1) > 0) & (hg(a, a1) > 0)),
                half=True,
                family=half,
            )
        )
        add(
            Market(
                f"{half}btts",
                "no",
                None,
                f"{half}btts",
                f"{name}: not both score",
                (lambda h, a, h1, a1, hg=hg: (hg(h, h1) == 0) | (hg(a, a1) == 0)),
                half=True,
                family=half,
            )
        )
    short = {"home": "H", "draw": "D", "away": "A"}
    word = {"home": "{h}", "draw": "Draw", "away": "{a}"}
    for r1 in res:
        for r2 in res:
            sel = f"{short[r1]}/{short[r2]}"
            add(
                Market(
                    "htft",
                    sel,
                    None,
                    f"htft_{sel}",
                    f"Half-time {word[r1]} / full-time {word[r2]}",
                    (lambda h, a, h1, a1, r1=r1, r2=r2: res[r1](h1, a1) & res[r2](h, a)),
                    half=True,
                    family="htft",
                )
            )
    add(
        Market(
            "hsh",
            "first",
            None,
            "hsh_first",
            "More goals in the 1st half",
            (lambda h, a, h1, a1: h1 + a1 > (h - h1) + (a - a1)),
            half=True,
            family="halves",
        )
    )
    add(
        Market(
            "hsh",
            "second",
            None,
            "hsh_second",
            "More goals in the 2nd half",
            (lambda h, a, h1, a1: h1 + a1 < (h - h1) + (a - a1)),
            half=True,
            family="halves",
        )
    )
    add(
        Market(
            "hsh",
            "equal",
            None,
            "hsh_equal",
            "Same goals in each half",
            (lambda h, a, h1, a1: h1 + a1 == (h - h1) + (a - a1)),
            half=True,
            family="halves",
        )
    )
    for side, home in (("home", True), ("away", False)):
        t = "{h}" if home else "{a}"
        add(
            Market(
                f"bh_{side}",
                "yes",
                None,
                f"bh_{side}",
                f"{t} score in both halves",
                (
                    lambda h, a, h1, a1, hm=home: (
                        ((h1 if hm else a1) > 0) & (((h - h1) if hm else (a - a1)) > 0)
                    )
                ),
                half=True,
                family="halves",
            )
        )
        add(
            Market(
                f"bh_{side}",
                "no",
                None,
                f"bh_{side}",
                f"{t} not score in both halves",
                (
                    lambda h, a, h1, a1, hm=home: (
                        ((h1 if hm else a1) == 0) | (((h - h1) if hm else (a - a1)) == 0)
                    )
                ),
                half=True,
                family="halves",
            )
        )
    return out


CATALOGUE = catalogue()
REGISTRY = {(m.market, m.sel, m.line): m for m in CATALOGUE}
NEW_MARKETS = tuple(dict.fromkeys(m.market for m in CATALOGUE))
VOID_MARKETS = ("dnb", "ahc", "atot")
HALF_MARKETS = tuple(dict.fromkeys(m.market for m in CATALOGUE if m.half))


def find(market: str, sel: str, line: float | None) -> Market | None:
    line = None if line is None or (isinstance(line, float) and np.isnan(line)) else float(line)
    return REGISTRY.get((market, sel, line))


# -- half model -----------------------------------------------------------------------------
def half_shares(
    matches: pd.DataFrame, before_season: int | None = None
) -> dict[str, tuple[float, float]]:
    """League -> (share of home goals, share of away goals) scored before half-time, from
    seasons before `before_season` (all seasons if None)."""
    m = matches.dropna(subset=["HTHG", "HTAG", "FTHG", "FTAG"])
    if before_season is not None:
        m = m[m["season"] < before_season]
    out = {}
    for lg, g in m.groupby("league"):
        out[str(lg)] = (
            float(g["HTHG"].sum() / max(g["FTHG"].sum(), 1)),
            float(g["HTAG"].sum() / max(g["FTAG"].sum(), 1)),
        )
    return out


def _thin(n: int, share: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(len(share), n, n): P(first-half goals = k | full-time goals = g) = Bin(k; g, share)."""
    g = np.arange(n)
    out: npt.NDArray[np.float64] = binom.pmf(
        g[None, None, :], g[None, :, None], share[:, None, None]
    )
    return out


def joint(
    grids: npt.NDArray[np.float64],
    s_home: float | npt.NDArray[np.float64],
    s_away: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Full-time grids (n, G, G) or one (G, G) -> P(h, a, h1, a1), shape (n, G, G, G, G)."""
    one = grids.ndim == 2
    g = grids[None] if one else grids
    n, size = g.shape[0], g.shape[1]
    sh = np.broadcast_to(np.asarray(s_home, dtype=np.float64), (n,))
    sa = np.broadcast_to(np.asarray(s_away, dtype=np.float64), (n,))
    bh, ba = _thin(size, sh), _thin(size, sa)
    out = g[:, :, :, None, None] * bh[:, :, None, :, None] * ba[:, None, :, None, :]
    return out[0] if one else out


def masks(
    markets: list[Market], size: int, half: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(win mask, void mask) per market as float arrays (cells, markets) over a 2-D grid
    (full-time markets) or a 4-D joint grid (half markets)."""
    if half:
        h, a, h1, a1 = np.indices((size,) * 4)
    else:
        h, a = np.indices((size, size))
        h1 = a1 = np.zeros_like(h)
    w = np.stack([np.asarray(m.win(h, a, h1, a1), dtype=bool).ravel() for m in markets], axis=1)
    v = np.stack(
        [
            np.zeros(h.size, bool)
            if m.void is None
            else np.asarray(m.void(h, a, h1, a1), dtype=bool).ravel()
            for m in markets
        ],
        axis=1,
    )
    return w.astype(np.float64), v.astype(np.float64)


def chances(
    grids: npt.NDArray[np.float64],
    markets: list[Market],
    shares: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    chunk: int = 400,
) -> npt.NDArray[np.float64]:
    """(n, markets): each market's chance for each grid; refund markets: P(win | no refund).
    Half markets need `shares` (per-row home and away shares); otherwise they come out NaN."""
    g = grids[None] if grids.ndim == 2 else grids
    n, size = g.shape[0], g.shape[1]
    out = np.full((n, len(markets)), np.nan)
    ft = [i for i, m in enumerate(markets) if not m.half]
    hf = [i for i, m in enumerate(markets) if m.half]
    if ft:
        w, v = masks([markets[i] for i in ft], size, False)
        flat = g.reshape(n, -1)
        pw, pv = flat @ w, flat @ v
        out[:, ft] = pw / np.clip(1 - pv, 1e-12, None)
    if hf and shares is not None:
        w, _ = masks([markets[i] for i in hf], size, True)
        sh, sa = (np.broadcast_to(np.asarray(s, dtype=np.float64), (n,)) for s in shares)
        for i in range(0, n, chunk):
            j = joint(g[i : i + chunk], sh[i : i + chunk], sa[i : i + chunk])
            out[i : i + chunk, hf] = j.reshape(len(j), -1) @ w
    return out


def outcome(m: Market, h: int, a: int, h1: float | None, a1: float | None) -> bool | None:
    """Graded result of one bet: True / False, None when refunded or half-time is unknown."""
    if m.half and (h1 is None or a1 is None or pd.isna(h1) or pd.isna(a1)):
        return None
    args = [np.array([x]) for x in (h, a, int(h1 or 0), int(a1 or 0))]
    if m.void is not None and bool(m.void(*args)[0]):
        return None
    return bool(m.win(*args)[0])
