"""Model vs early over/under 2.5 prices.

Rules pre-registered in docs/adr/0009-early-totals-test.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_edge.data.loader import read_raw_csv
from football_edge.markets.price_edge import bootstrap_roi

EARLY_COLS = {
    "ps_over": "P>2.5", "ps_under": "P<2.5", "b365_over": "B365>2.5", "b365_under": "B365<2.5",
    "max_over": "Max>2.5", "max_under": "Max<2.5", "psc_over": "PC>2.5", "psc_under": "PC<2.5",
}  # fmt: skip


@dataclass(frozen=True)
class Variant:
    name: str
    price: str  # "b365" or "max"
    threshold: float


VARIANTS = (Variant("b365_3", "b365", 0.03), Variant("b365_6", "b365", 0.06),
            Variant("max_3", "max", 0.03), Variant("max_6", "max", 0.06))  # fmt: skip


def load_early_prices(raw_dir: Path, leagues: list[str], seasons: tuple[int, int]) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        for s in range(seasons[0], seasons[1] + 1):
            path = raw_dir / f"{lg}_{s}.csv"
            if not path.exists():
                continue
            df = read_raw_csv(path)
            t = pd.DataFrame({
                "league": lg,
                "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.normalize(),
                "home": df["HomeTeam"].str.strip(), "away": df["AwayTeam"].str.strip(),
            })  # fmt: skip
            for out, col in EARLY_COLS.items():
                v = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
                t[out] = v.where(v > 1.0) if isinstance(v, pd.Series) else v  # 0 = missing
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _fair_over(over: pd.Series, under: pd.Series) -> npt.NDArray[np.float64]:
    o, u = 1 / over.to_numpy(dtype=float), 1 / under.to_numpy(dtype=float)
    return np.asarray(o / (o + u), dtype=float)


def join(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    m = p.merge(prices, on=["league", "date", "home", "away"], how="inner")
    m["over"] = (m["goals_home"] + m["goals_away"]) >= 3
    m["fair_early"] = _fair_over(m["ps_over"], m["ps_under"])
    m["fair_close"] = _fair_over(m["psc_over"], m["psc_under"])
    return m


def _ll(p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(-(y * np.log(p) + (1 - y) * np.log(1 - p)), dtype=float)


def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)), dtype=float)


def test_a(m: pd.DataFrame, level: float, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Log-loss of early market, model and the out-of-sample blend (beta from earlier seasons)."""
    d = m.dropna(subset=["fair_early"]).sort_values("date").reset_index(drop=True)
    y = d["over"].to_numpy(dtype=float)
    mk, md = d["fair_early"].to_numpy(), d["p_over25"].to_numpy()
    blend = np.full(len(d), np.nan)
    grid = np.linspace(-1, 2, 301)
    for s in sorted(d["season"].unique()):
        past, now = (d["season"] < s).to_numpy(), (d["season"] == s).to_numpy()
        if past.sum() < 500:
            continue
        lp = [
            _ll(1 / (1 + np.exp(-(_logit(mk[past]) + b * (_logit(md[past]) - _logit(mk[past]))))),
                y[past]).mean()
            for b in grid
        ]  # fmt: skip
        b = grid[int(np.argmin(lp))]
        blend[now] = 1 / (1 + np.exp(-(_logit(mk[now]) + b * (_logit(md[now]) - _logit(mk[now])))))
    ok = ~np.isnan(blend)
    gain = _ll(mk[ok], y[ok]) - _ll(blend[ok], y[ok])  # positive = blend better than market
    rng = np.random.default_rng(seed)
    week = d.loc[ok, "date"] - pd.to_timedelta(d.loc[ok, "date"].dt.day_of_week, unit="D")
    blocks = pd.factorize(d.loc[ok, "league"].astype(str) + "|" + week.astype(str))[0]
    groups = [np.flatnonzero(blocks == k) for k in range(blocks.max() + 1)]
    boots = [gain[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])]
             .mean() for _ in range(n_boot)]  # fmt: skip
    a = (1 - level) / 2
    return {
        "n": int(ok.sum()), "market": float(_ll(mk[ok], y[ok]).mean()),
        "model": float(_ll(md[ok], y[ok]).mean()), "blend": float(_ll(blend[ok], y[ok]).mean()),
        "gain": float(gain.mean()), "lo": float(np.quantile(boots, a)),
        "hi": float(np.quantile(boots, 1 - a)),
    }  # fmt: skip


@dataclass
class Result:
    variant: Variant
    bets: pd.DataFrame
    roi: float = 0.0
    ci: tuple[float, float] = (0.0, 0.0)
    by_season: dict[int, float] = field(default_factory=dict)
    clv: float = float("nan")
    criteria: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(self.criteria.values())


def run_variant(m: pd.DataFrame, v: Variant, level: float, n_boot: int = 2000) -> Result:
    d = m.dropna(subset=[f"{v.price}_over", f"{v.price}_under"]).reset_index(drop=True)
    po, pu = d[f"{v.price}_over"].to_numpy(float), d[f"{v.price}_under"].to_numpy(float)
    p = d["p_over25"].to_numpy(float)
    e_over, e_under = p * po - 1, (1 - p) * pu - 1
    take_over = e_over >= e_under
    edge = np.where(take_over, e_over, e_under)
    odds = np.where(take_over, po, pu)
    won = np.where(take_over, d["over"], ~d["over"])
    close_fair = np.where(take_over, d["fair_close"], 1 - d["fair_close"])
    bets = pd.DataFrame({
        "league": d["league"], "season": d["season"], "date": d["date"],
        "home": d["home"], "away": d["away"], "side": np.where(take_over, "over", "under"),
        "model_p": np.where(take_over, p, 1 - p), "odds": odds, "edge": edge, "won": won,
        "stake": 100.0, "profit": np.where(won, 100.0 * (odds - 1), -100.0),
        "clv": odds * close_fair - 1,
    })[edge >= v.threshold].reset_index(drop=True)  # fmt: skip
    r = Result(v, bets)
    if bets.empty:
        r.criteria = {"at least 200 bets": False}
        return r
    r.roi = float(bets["profit"].sum() / bets["stake"].sum())
    r.ci = bootstrap_roi(bets, n_boot, level)
    r.by_season = {int(s): float(g["profit"].sum() / g["stake"].sum())
                   for s, g in bets.groupby("season")}  # fmt: skip
    r.clv = float(bets["clv"].mean())
    r.criteria = {
        "ROI interval above 0": r.roi > 0 and r.ci[0] > 0,
        "ROI > 0 in >= 4 of 6 seasons": sum(x > 0 for x in r.by_season.values()) >= 4,
        "at least 200 bets": len(bets) >= 200,
    }
    return r


def render(a: dict[str, float], results: list[Result], level: float, matched: int) -> str:
    lines = [
        "# Early over/under 2.5 prices vs the model",
        "",
        f"Rules pre-registered in docs/adr/0009-early-totals-test.md. Seasons 2020/21-2025/26, "
        f"{matched:,} matches with model predictions and early prices. Intervals {level:.2%}.",
        "",
        "## Test A: does the model add to the early Pinnacle price?",
        "",
        "| Forecast | Log-loss |",
        "|---|---|",
        f"| Early market alone (Pinnacle, margin removed) | {a['market']:.4f} |",
        f"| Model alone | {a['model']:.4f} |",
        f"| Blend (beta from earlier seasons) | {a['blend']:.4f} |",
        "",
        f"Blend minus market: {a['gain']:+.5f} per match "
        f"(interval {a['lo']:+.5f} to {a['hi']:+.5f}), {a['n']:,} matches.",
        "",
        "## Test B: betting the model at early prices (flat 100)",
        "",
        "| Variant | Bets | Staked | Profit | ROI | ROI interval | Win rate | Avg odds | CLV "
        "| Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        b = r.bets
        if b.empty:
            lines.append(f"| {r.variant.name} | 0 | 0 | 0 | - | - | - | - | - | FAIL |")
            continue
        lines.append(
            f"| {r.variant.name} | {len(b):,} | {b['stake'].sum():,.0f} "
            f"| {b['profit'].sum():+,.0f} "
            f"| {r.roi:+.2%} | {r.ci[0]:+.2%} to {r.ci[1]:+.2%} | {b['won'].mean():.1%} "
            f"| {b['odds'].mean():.2f} | {r.clv:+.2%} | {'PASS' if r.passed else 'FAIL'} |"
        )
    for r in results:
        lines += ["", f"### {r.variant.name}", "", "| Criterion | Result |", "|---|---|"]
        lines += [f"| {k} | {'PASS' if ok else 'FAIL'} |" for k, ok in r.criteria.items()]
        lines += ["", "| Season | ROI |", "|---|---|"]
        lines += [f"| {s} | {x:+.2%} |" for s, x in r.by_season.items()]
    return "\n".join(lines) + "\n"
