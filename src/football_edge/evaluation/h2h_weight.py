"""Head-to-head blended into expected goals with a chosen weight.

Rules pre-registered in docs/adr/0010-h2h-weight-test.md.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import poisson

FloatArr = npt.NDArray[np.float64]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
MAX_GOALS = 10


def h2h_rates(preds: pd.DataFrame, history: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """Per prediction row: meetings in the window, and venue-adjusted head-to-head goal rates
    (with one pseudo-meeting at the model's own rate)."""
    h = history.sort_values("date").reset_index(drop=True)
    h["date"] = pd.to_datetime(h["date"])
    # League home/away goals ratio from matches before each date (expanding).
    ratio_at: dict[str, tuple[npt.NDArray[np.datetime64], FloatArr]] = {}
    for lg, g in h.groupby("league"):
        cum_h, cum_a = g["FTHG"].cumsum().to_numpy(float), g["FTAG"].cumsum().to_numpy(float)
        ratio_at[str(lg)] = (
            g["date"].to_numpy(),
            np.where(cum_a > 0, cum_h / np.maximum(cum_a, 1), 1.3),
        )
    pairs: dict[frozenset[str], list[tuple[pd.Timestamp, str, int, int]]] = defaultdict(list)
    h_dates = list(pd.to_datetime(h["date"]))
    h_home, h_away = h["home"].astype(str).tolist(), h["away"].astype(str).tolist()
    h_hg, h_ag = h["FTHG"].astype(int).tolist(), h["FTAG"].astype(int).tolist()
    for md, mh, ma, hg, ag in zip(h_dates, h_home, h_away, h_hg, h_ag, strict=True):
        pairs[frozenset((mh, ma))].append((md, mh, hg, ag))

    out = []
    p_dates = list(pd.to_datetime(preds["date"]))
    p_lg = preds["league"].astype(str).tolist()
    p_home, p_away = preds["home"].astype(str).tolist(), preds["away"].astype(str).tolist()
    p_eh, p_ea = preds["exp_home"].astype(float).tolist(), preds["exp_away"].astype(float).tolist()
    for d, lg, home, away, eh, ea in zip(p_dates, p_lg, p_home, p_away, p_eh, p_ea, strict=True):
        dates, ratios = ratio_at[lg]
        k = int(np.searchsorted(dates, np.datetime64(d), side="left")) - 1
        ratio = float(ratios[k]) if k >= 0 else 1.3
        lo = d - pd.DateOffset(years=years)
        gh = ga = 0.0
        n = 0
        for md, mh, hg, ag in pairs.get(frozenset((home, away)), []):
            if not lo <= md < d:
                continue
            n += 1
            if mh == home:  # same venue as now
                gh, ga = gh + hg, ga + ag
            else:  # roles reversed: our home team was the away side then
                gh, ga = gh + ag * ratio, ga + hg / ratio
        out.append(
            {"meetings": n, "rate_home": (gh + eh) / (n + 1), "rate_away": (ga + ea) / (n + 1)}
        )
    return pd.DataFrame(out, index=preds.index)


def score_probs(lh: FloatArr, la: FloatArr) -> npt.NDArray[np.float64]:
    """(n, 11, 11) independent-Poisson score grid, renormalised."""
    g = np.arange(MAX_GOALS + 1)
    ph, pa = poisson.pmf(g[None, :], lh[:, None]), poisson.pmf(g[None, :], la[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    return np.asarray(grid / grid.sum(axis=(1, 2), keepdims=True), dtype=float)


def losses(preds: pd.DataFrame, rates: pd.DataFrame, w: float) -> dict[str, FloatArr]:
    """Per-match log-loss (exact score, 1x2, ou25) with weight w."""
    lh0, la0 = preds["exp_home"].to_numpy(float), preds["exp_away"].to_numpy(float)
    has = (rates["meetings"] > 0).to_numpy()
    lh = np.where(has, lh0 ** (1 - w) * rates["rate_home"].to_numpy(float) ** w, lh0)
    la = np.where(has, la0 ** (1 - w) * rates["rate_away"].to_numpy(float) ** w, la0)
    grid = score_probs(lh, la)
    hg = np.minimum(preds["goals_home"].to_numpy(int), MAX_GOALS)
    ag = np.minimum(preds["goals_away"].to_numpy(int), MAX_GOALS)
    idx = np.arange(len(preds))
    eps = 1e-12
    exact = -np.log(grid[idx, hg, ag] + eps)
    i, j = np.indices((MAX_GOALS + 1, MAX_GOALS + 1))
    p_h = (grid * (i > j)).sum(axis=(1, 2))
    p_d = (grid * (i == j)).sum(axis=(1, 2))
    p_a = 1 - p_h - p_d
    res = np.select([hg > ag, hg == ag], [p_h, p_d], p_a)
    over = (grid * (i + j >= 3)).sum(axis=(1, 2))
    tot = hg + ag
    ou = np.where(tot >= 3, over, 1 - over)
    return {"exact": exact, "1x2": -np.log(res + eps), "ou25": -np.log(ou + eps)}


def _blocks(preds: pd.DataFrame) -> npt.NDArray[np.intp]:
    d = pd.to_datetime(preds["date"])
    week = d - pd.to_timedelta(d.dt.day_of_week, unit="D")
    return np.asarray(pd.factorize(preds["league"].astype(str) + "|" + week.astype(str))[0])


def boot_ci(diff: FloatArr, blocks: npt.NDArray[np.intp], level: float = 0.95,
            n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:  # fmt: skip
    groups = [np.flatnonzero(blocks == b) for b in range(blocks.max() + 1)]
    sums = np.array([diff[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), (n_boot, len(groups)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def run(preds: pd.DataFrame, history: pd.DataFrame, years: int = 3) -> str:
    preds = preds.sort_values("date").reset_index(drop=True)
    rates = h2h_rates(preds, history, years)
    by_w = {float(w): losses(preds, rates, float(w)) for w in GRID}
    seasons = preds["season"].to_numpy()
    test = (seasons >= 2020) & (seasons <= 2025)
    chosen: dict[int, float] = {}
    walk = {k: np.full(len(preds), np.nan) for k in ("exact", "1x2", "ou25")}
    for s in range(2020, 2026):
        past, now = seasons < s, seasons == s
        w = min(GRID, key=lambda x: by_w[float(x)]["exact"][past].mean())
        chosen[s] = float(w)
        for k in walk:
            walk[k][now] = by_w[float(w)][k][now]
    base = by_w[0.0]
    blocks = _blocks(preds[test])
    has = (rates["meetings"] > 0).to_numpy()

    lines = [
        "# Head-to-head (past 3 years) with a chosen weight",
        "",
        "Rules pre-registered in docs/adr/0010-h2h-weight-test.md. Test seasons 2020/21-2025/26, "
        f"{int(test.sum()):,} matches; {has[test].mean():.0%} of them had at least one meeting in "
        f"the 3 years before (average {rates['meetings'][test].mean():.1f} meetings).",
        "",
        "## Weight chosen each season (from earlier seasons only)",
        "",
        "| Season | w |",
        "|---|---|",
        *[f"| {s} | {w:.2f} |" for s, w in chosen.items()],
        "",
        "## Result (walk-forward weight vs no head-to-head)",
        "",
        "| Measure | Without h2h | With h2h | Improvement per match | 95% interval | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for k, label in (("exact", "Exact final score"), ("1x2", "1X2"), ("ou25", "Over/under 2.5")):
        diff = base[k][test] - walk[k][test]
        lo, hi = boot_ci(diff, blocks)
        verdict = "HELPS" if diff.mean() > 0 and lo > 0 else "DOES NOT HELP"
        if k != "exact":
            verdict = f"({verdict.lower()}; reported only)"
        lines.append(
            f"| {label} | {base[k][test].mean():.4f} | {walk[k][test].mean():.4f} "
            f"| {diff.mean():+.5f} | {lo:+.5f} to {hi:+.5f} | {verdict} |"
        )
    lines += ["", "## Every weight on the test seasons (in-sample, for information only)", "",
              "| w | Exact score | 1X2 | Over/under 2.5 |", "|---|---|---|---|"]  # fmt: skip
    for w in GRID:
        lw = by_w[float(w)]
        lines.append(f"| {w:.2f} | {lw['exact'][test].mean():.4f} | {lw['1x2'][test].mean():.4f} "
                     f"| {lw['ou25'][test].mean():.4f} |")  # fmt: skip
    return "\n".join(lines) + "\n"
