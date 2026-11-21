"""Paired comparison of two models on the same matches (e.g. with vs without head-to-head)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArr = npt.NDArray[np.float64]


def market_columns(preds: pd.DataFrame, market: str) -> tuple[FloatArr, npt.NDArray[np.intp]]:
    """Model probabilities and actual outcome for a market: '1x2' or 'ou25' (over, under)."""
    if market == "1x2":
        return preds[["p_h", "p_d", "p_a"]].to_numpy(dtype=float), preds["outcome"].to_numpy()
    if market == "ou25":
        if "p_over25" not in preds:
            raise ValueError("these predictions have no over/under probabilities")
        over = preds["p_over25"].to_numpy(dtype=float)
        under_happened = (preds["goals_home"] + preds["goals_away"]) < 3
        return np.c_[over, 1 - over], np.asarray(under_happened, dtype=np.intp)
    raise ValueError(f"unknown market {market!r} (use 1x2 or ou25)")


def per_match_log_loss(p: FloatArr, y: npt.NDArray[np.intp]) -> FloatArr:
    picked = p[np.arange(len(y)), y]
    return np.asarray(-np.log(np.clip(picked, 1e-15, 1.0)), dtype=np.float64)


@dataclass
class Comparison:
    market: str
    n: int
    log_loss_a: float
    log_loss_b: float
    improvement: float  # a minus b: positive means b is better
    ci: tuple[float, float]
    by_season: dict[int, float]
    by_league: dict[str, float]

    @property
    def b_helps(self) -> bool:
        return self.improvement > 0 and self.ci[0] > 0


def paired_compare(
    a: pd.DataFrame,
    b: pd.DataFrame,
    market: str,
    seasons: list[int],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Per-match log-loss of model a minus model b on the same test matches."""
    m = a.merge(b, on="row", suffixes=("_a", "_b"))
    m = m[m["season_a"].isin(seasons)].reset_index(drop=True)

    def side(suffix: str) -> pd.DataFrame:
        cols = {c: c[: -len(suffix)] for c in m.columns if c.endswith(suffix)}
        return m[list(cols)].rename(columns=cols)

    pa, ya = market_columns(side("_a"), market)
    pb, yb = market_columns(side("_b"), market)
    if not (ya == yb).all():
        raise ValueError("the two prediction files disagree on outcomes")
    d = per_match_log_loss(pa, ya) - per_match_log_loss(pb, yb)

    keys = m["league_a"].astype(str) + "|" + m["snapshot_a"].astype(str)
    blocks = np.asarray(pd.factorize(keys)[0])
    order = np.argsort(blocks, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(blocks[order])) + 1)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            d[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean()
            for _ in range(n_boot)
        ]
    )
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return Comparison(
        market=market,
        n=len(d),
        log_loss_a=float(per_match_log_loss(pa, ya).mean()),
        log_loss_b=float(per_match_log_loss(pb, yb).mean()),
        improvement=float(d.mean()),
        ci=(float(np.quantile(boot, lo)), float(np.quantile(boot, hi))),
        by_season={int(s): float(d[(m.season_a == s).to_numpy()].mean()) for s in seasons},
        by_league={
            str(lg): float(d[(m.league_a == lg).to_numpy()].mean())
            for lg in sorted(m.league_a.unique())
        },
    )
