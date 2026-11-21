"""Price-comparison backtest: bet only when the best price beats the sharp fair price.

No prediction model is involved. Rules are pre-registered in docs/adr/0004-price-edge-test.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_edge.markets.margin import remove_margin
from football_edge.models.base import outcome_index

FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Variant:
    name: str
    fair_cols: tuple[str, str, str]  # sharp odds the fair probability comes from
    fair_src_col: str  # column naming the source of those odds
    price_cols: tuple[str, str, str]  # odds we bet at
    threshold: float


VARIANTS = (
    Variant("pre_2", ("pre_h", "pre_d", "pre_a"), "pre_src", ("max_h", "max_d", "max_a"), 0.02),
    Variant("pre_5", ("pre_h", "pre_d", "pre_a"), "pre_src", ("max_h", "max_d", "max_a"), 0.05),
    Variant(
        "close_2",
        ("close_h", "close_d", "close_a"),
        "close_src",
        ("maxc_h", "maxc_d", "maxc_a"),
        0.02,
    ),
    Variant(
        "close_5",
        ("close_h", "close_d", "close_a"),
        "close_src",
        ("maxc_h", "maxc_d", "maxc_a"),
        0.05,
    ),
)


def select_bets(matches: pd.DataFrame, v: Variant, stake: float = 100.0) -> pd.DataFrame:
    """One row per bet placed under variant `v`, with its profit."""
    ok = (matches[v.fair_src_col] == "PS") & matches[list(v.price_cols)].notna().all(axis=1)
    m = matches[ok].reset_index(drop=True)
    if m.empty:
        return pd.DataFrame()
    fair = remove_margin(m[list(v.fair_cols)].to_numpy(dtype=float))
    price = m[list(v.price_cols)].to_numpy(dtype=float)
    edge = price * fair - 1
    pick = edge.argmax(axis=1)
    rows = np.arange(len(m))
    best_edge = edge[rows, pick]
    take = best_edge >= v.threshold
    won = outcome_index(m["FTR"]) == pick
    odds_taken = price[rows, pick]
    bets = pd.DataFrame(
        {
            "league": m["league"],
            "season": m["season"],
            "date": m["date"],
            "home": m["home"],
            "away": m["away"],
            "pick": np.array(["H", "D", "A"])[pick],
            "odds": odds_taken,
            "fair_prob": fair[rows, pick],
            "edge": best_edge,
            "won": won,
            "stake": stake,
            "profit": np.where(won, stake * (odds_taken - 1), -stake),
        }
    )[take].reset_index(drop=True)
    if v.fair_src_col == "pre_src":  # closing line value against Pinnacle's closing price
        close_ok = (m["close_src"] == "PS").to_numpy()
        close_fair = np.full_like(fair, np.nan)
        close_fair[close_ok] = remove_margin(
            m.loc[close_ok, ["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
        )
        bets["clv"] = (odds_taken * close_fair[rows, pick] - 1)[take]
    return bets


def _blocks(bets: pd.DataFrame) -> npt.NDArray[np.intp]:
    week = bets["date"] - pd.to_timedelta(bets["date"].dt.day_of_week, unit="D")
    keys = bets["league"].astype(str) + "|" + week.astype(str)
    return np.asarray(pd.factorize(keys)[0], dtype=np.intp)


def bootstrap_roi(
    bets: pd.DataFrame, n_boot: int, level: float, seed: int = 0
) -> tuple[float, float]:
    blocks = _blocks(bets)
    order = np.argsort(blocks, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(blocks[order])) + 1)
    profit, stake = bets["profit"].to_numpy(), bets["stake"].to_numpy()
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        rois[i] = profit[idx].sum() / stake[idx].sum()
    a = (1 - level) / 2
    return float(np.quantile(rois, a)), float(np.quantile(rois, 1 - a))


@dataclass
class VariantResult:
    variant: Variant
    n_bets: int
    staked: float
    profit: float
    roi: float
    roi_ci: tuple[float, float]
    roi_by_season: dict[int, float]
    hit_rate: float
    avg_odds: float
    clv: float | None
    criteria: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.criteria.values())


def run_variant(
    matches: pd.DataFrame, v: Variant, level: float, n_boot: int = 2000
) -> tuple[VariantResult, pd.DataFrame]:
    bets = select_bets(matches, v)
    n = len(bets)
    if n == 0:
        res = VariantResult(v, 0, 0.0, 0.0, 0.0, (0.0, 0.0), {}, 0.0, 0.0, None)
        res.criteria = {"at least 200 bets": False}
        return res, bets
    staked, profit = float(bets["stake"].sum()), float(bets["profit"].sum())
    by_season = {
        int(str(s)): float(g["profit"].sum() / g["stake"].sum()) for s, g in bets.groupby("season")
    }
    res = VariantResult(
        variant=v,
        n_bets=n,
        staked=staked,
        profit=profit,
        roi=profit / staked,
        roi_ci=bootstrap_roi(bets, n_boot, level),
        roi_by_season=by_season,
        hit_rate=float(bets["won"].mean()),
        avg_odds=float(bets["odds"].mean()),
        clv=float(bets["clv"].mean()) if "clv" in bets and bets["clv"].notna().any() else None,
    )
    res.criteria = {
        "ROI interval above 0": res.roi > 0 and res.roi_ci[0] > 0,
        "ROI > 0 in >= 4 of 7 seasons": sum(r > 0 for r in by_season.values()) >= 4,
        "at least 200 bets": n >= 200,
    }
    return res, bets


def render_report(results: list[VariantResult], level: float, seasons: tuple[int, int]) -> str:
    tick = {True: "PASS", False: "FAIL"}
    lines = [
        "# Price-comparison test (no model)",
        "",
        "Rules pre-registered in docs/adr/0004-price-edge-test.md. "
        f"Seasons {seasons[0]}-{seasons[1]}. Intervals are {level:.2%} (Bonferroni, 4 variants).",
        "",
        "| Variant | Bets | Staked | Profit | ROI | ROI interval | Hit rate | Avg odds | CLV "
        "| Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        clv = f"{r.clv:+.2%}" if r.clv is not None else "n/a"
        lines.append(
            f"| {r.variant.name} | {r.n_bets} | {r.staked:,.0f} | {r.profit:+,.0f} | {r.roi:+.2%} "
            f"| {r.roi_ci[0]:+.2%} to {r.roi_ci[1]:+.2%} | {r.hit_rate:.1%} | {r.avg_odds:.2f} "
            f"| {clv} | {'PASS' if r.passed else 'FAIL'} |"
        )
    for r in results:
        lines += ["", f"## {r.variant.name}", "", "| Criterion | Result |", "|---|---|"]
        lines += [f"| {k} | {tick[ok]} |" for k, ok in r.criteria.items()]
        lines += ["", "| Season | ROI |", "|---|---|"]
        lines += [f"| {s} | {roi:+.2%} |" for s, roi in r.roi_by_season.items()]
    lines.append("")
    return "\n".join(lines)
