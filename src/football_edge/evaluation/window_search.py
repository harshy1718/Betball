"""Search every probability window (docs/adr/0012-window-search.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.markets.price_edge import bootstrap_roi

DISCOVERY, HOLDOUT = (2018, 2022), (2023, 2025)
MIN_BETS = 200


def candidates(
    m: pd.DataFrame, market: str, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Every outcome of every match as a possible bet: model probability, price, won.
    With `rng`, probabilities are shuffled between matches within league-season (placebo)."""
    hg, ag = m["goals_home"].to_numpy(int), m["goals_away"].to_numpy(int)
    if market == "1x2":
        probs = m[["p_h", "p_d", "p_a"]].to_numpy(float)
        prices = m[["b365_h", "b365_d", "b365_a"]].to_numpy(float)
        actual = np.select([hg > ag, hg == ag], [0, 1], 2)
        won = np.stack([actual == k for k in range(3)], axis=1)
    else:
        q = m["p_over25"].to_numpy(float)
        probs = np.stack([q, 1 - q], axis=1)
        prices = m[["b365_over", "b365_under"]].to_numpy(float)
        over = hg + ag >= 3
        won = np.stack([over, ~over], axis=1)
    if rng is not None:
        probs = probs.copy()
        for _, idx in m.reset_index(drop=True).groupby(["league", "season"]).indices.items():
            probs[idx] = probs[rng.permutation(idx)]
    k = probs.shape[1]
    out = pd.DataFrame({
        "league": np.repeat(m["league"].to_numpy(), k),
        "season": np.repeat(m["season"].to_numpy(), k),
        "date": np.repeat(m["date"].to_numpy(), k), "p": probs.ravel(), "odds": prices.ravel(),
        "won": won.ravel(),
    })  # fmt: skip
    out = out.dropna(subset=["odds"]).reset_index(drop=True)
    out["stake"] = 100.0
    out["profit"] = np.where(out["won"], 100.0 * (out["odds"] - 1), -100.0)
    return out


def scan(c: pd.DataFrame) -> pd.DataFrame:
    """ROI of every window [lo, hi) with >= MIN_BETS bets (fast: sorted prefix sums)."""
    c = c.sort_values("p")
    p, prof = c["p"].to_numpy(), c["profit"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(prof)])
    rows = []
    for lo in np.round(np.arange(0.20, 0.9001, 0.01), 2):
        i = int(np.searchsorted(p, lo, side="left"))
        for width in np.round(np.arange(0.02, 0.3001, 0.01), 2):
            hi = round(lo + width, 2)
            if hi > 1.0:
                break
            j = int(np.searchsorted(p, hi, side="left"))
            n = j - i
            if n >= MIN_BETS:
                rows.append((lo, hi, n, (cum[j] - cum[i]) / (100.0 * n)))
    return pd.DataFrame(rows, columns=["lo", "hi", "bets", "roi"])


def _in(c: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return c[(c["p"] >= lo) & (c["p"] < hi)]


def run(m: pd.DataFrame, seed: int = 0) -> str:
    disc = m["season"].between(*DISCOVERY)
    hold = m["season"].between(*HOLDOUT)
    names = {"1x2": "Match winner (home, draw, away)", "ou25": "Over/under 2.5 (over, under)"}
    lines = [
        "# Every probability window, searched (model only, Bet365 prices)",
        "",
        "Rules pre-registered in docs/adr/0012-window-search.md. Search on 2018/19-2022/23 "
        f"({int(disc.sum()):,} matches), test on 2023/24-2025/26 ({int(hold.sum()):,} matches, "
        "never seen by the search). Windows need 200+ bets. Flat 100.",
    ]
    for market in ("1x2", "ou25"):
        c_disc, c_hold = candidates(m[disc], market), candidates(m[hold], market)
        grid = scan(c_disc).sort_values("roi", ascending=False)
        rng = np.random.default_rng(seed)
        placebo = scan(candidates(m[disc], market, rng)).sort_values("roi", ascending=False)
        best = grid.iloc[0]
        b = _in(c_hold, best["lo"], best["hi"]).reset_index(drop=True)
        lo_ci, hi_ci = bootstrap_roi(b, 2000, 0.975) if len(b) else (np.nan, np.nan)
        roi = float(b["profit"].sum() / b["stake"].sum()) if len(b) else np.nan
        ok = len(b) > 0 and roi > 0 and lo_ci > 0
        lines += [
            "",
            f"## {names[market]}",
            "",
            f"Windows searched: {len(grid):,}. Profitable in the search seasons: "
            f"**{int((grid['roi'] > 0).sum()):,}** ({(grid['roi'] > 0).mean():.0%}). "
            f"With the model's probabilities shuffled (a model that knows nothing): "
            f"**{int((placebo['roi'] > 0).sum()):,}** ({(placebo['roi'] > 0).mean():.0%}).",
            "",
            f"**Best window found: {best['lo']:.0%} to {best['hi']:.0%}**, "
            f"{int(best['bets']):,} bets, ROI {best['roi']:+.2%} in the search seasons.",
            "",
            f"**Same window on the 3 unseen seasons:** {len(b):,} bets, won {b['won'].mean():.1%}, "
            f"avg odds {b['odds'].mean():.2f}, profit {b['profit'].sum():+,.0f}, ROI {roi:+.2%} "
            f"(interval {lo_ci:+.2%} to {hi_ci:+.2%}). **{'MAKES MONEY' if ok else 'DOES NOT'}.**",
            "",
            "Top 10 windows from the search, and what each did on the unseen seasons:",
            "",
            "| Window | Bets (search) | ROI (search) | Bets (unseen) | ROI (unseen) |",
            "|---|---|---|---|---|",
        ]
        top = grid.head(10)
        cols = [top[c].tolist() for c in ("lo", "hi", "bets", "roi")]
        for wlo, whi, wn, wroi in zip(*cols, strict=True):
            h = _in(c_hold, float(wlo), float(whi))
            hr = h["profit"].sum() / h["stake"].sum() if len(h) else np.nan
            lines.append(f"| {wlo:.0%}-{whi:.0%} | {int(wn):,} | {wroi:+.2%} | {len(h):,} "
                         f"| {hr:+.2%} |")  # fmt: skip
        pb = placebo.iloc[0]
        lines += ["", f"Luck control: the shuffled model's best window was {pb['lo']:.0%}-"
                  f"{pb['hi']:.0%} with ROI {pb['roi']:+.2%} in the search seasons."]  # fmt: skip
    return "\n".join(lines) + "\n"
