"""The stack test: does a model add information beyond the bookmaker's odds?

The blend moves the market's probabilities towards the model's by a factor beta:

    q_k  is proportional to  p_market_k * (p_model_k / p_market_k) ** beta

beta = 0 is the market alone; beta = 1 is the model alone. If the best beta on
unseen seasons is clearly above 0 and the blend beats the market, the model knows
something the odds don't. Pre-registered criteria: docs/adr/0003-kill-rule.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize_scalar

from football_edge.evaluation.metrics import brier_score, log_loss

Probs = npt.NDArray[np.float64]

BETA_BOUNDS = (-1.0, 3.0)
EPS = 1e-12


def blend(p_market: Probs, p_model: Probs, beta: float) -> Probs:
    """Market moved towards the model by `beta` (0 = market, 1 = model). Written by Harsh."""
    if p_market.shape != p_model.shape or p_market.ndim != 2 or p_market.shape[1] < 2:
        raise ValueError(
            f"shapes must match and be (n, k >= 2): {p_market.shape} vs {p_model.shape}"
        )
    raw = p_market * ((p_model / p_market) ** beta)
    blend = raw / (raw.sum(axis=1, keepdims=True))
    return blend


def fit_beta(p_market: Probs, p_model: Probs, outcomes: npt.NDArray[np.intp]) -> float:
    """The beta that minimises the blend's log-loss on these matches."""
    res = minimize_scalar(
        lambda b: log_loss(blend(p_market, p_model, b), outcomes),
        bounds=BETA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


def per_match_log_loss(p: Probs, outcomes: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
    picked = p[np.arange(len(outcomes)), outcomes]
    return np.asarray(-np.log(np.clip(picked, 1e-15, 1.0)), dtype=np.float64)


@dataclass
class StackResult:
    eval_seasons: list[int]
    beta_by_season: dict[int, float]
    delta: float  # market log-loss minus blend log-loss, per match (positive = model helps)
    delta_ci: tuple[float, float]
    beta_pooled: float
    beta_ci: tuple[float, float]
    brier_market: float
    brier_blend: float
    delta_by_season: dict[int, float]
    delta_by_league: dict[str, float]
    log_loss_market: float
    log_loss_blend: float
    log_loss_model: float
    n: int
    level: float = 0.95
    criteria: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.criteria.values())


def walk_forward_blend(
    frame: pd.DataFrame, p_market: Probs, p_model: Probs, eval_seasons: list[int]
) -> tuple[Probs, dict[int, float]]:
    """Blend each evaluation season using a beta fitted only on earlier seasons."""
    seasons = frame["season"].to_numpy()
    y = frame["outcome"].to_numpy()
    q = np.full_like(p_market, np.nan)
    betas: dict[int, float] = {}
    for s in eval_seasons:
        past = seasons < s
        if not past.any():
            raise ValueError(f"no earlier seasons to fit beta for season {s}")
        betas[s] = fit_beta(p_market[past], p_model[past], y[past])
        now = seasons == s
        q[now] = blend(p_market[now], p_model[now], betas[s])
    return q, betas


def _block_ids(frame: pd.DataFrame) -> npt.NDArray[np.intp]:
    keys = frame["league"].astype(str) + "|" + frame["snapshot"].astype(str)
    return np.asarray(pd.factorize(keys)[0], dtype=np.intp)


def _bootstrap_rows(
    blocks: npt.NDArray[np.intp], n_boot: int, rng: np.random.Generator
) -> list[npt.NDArray[np.intp]]:
    order = np.argsort(blocks, kind="stable")
    bounds = np.flatnonzero(np.diff(blocks[order])) + 1
    groups = np.split(order, bounds)
    n_blocks = len(groups)
    return [
        np.concatenate([groups[i] for i in rng.integers(0, n_blocks, n_blocks)])
        for _ in range(n_boot)
    ]


def stack_test(
    frame: pd.DataFrame,
    p_market: Probs,
    p_model: Probs,
    eval_seasons: list[int],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> StackResult:
    """Run the pre-registered stack test. `frame` needs season, league, snapshot, outcome."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    q_all, betas = walk_forward_blend(frame, p_market, p_model, eval_seasons)
    ev = frame["season"].isin(eval_seasons).to_numpy()
    f_ev = frame[ev].reset_index(drop=True)
    y = f_ev["outcome"].to_numpy()
    pm, pq, pmod = p_market[ev], q_all[ev], p_model[ev]

    d = per_match_log_loss(pm, y) - per_match_log_loss(pq, y)
    rng = np.random.default_rng(seed)
    samples = _bootstrap_rows(_block_ids(f_ev), n_boot, rng)
    boot_delta = np.array([d[r].mean() for r in samples])

    # Pooled beta over every season with predictions, with a block-bootstrap interval.
    y_all = frame["outcome"].to_numpy()
    beta_pooled = fit_beta(p_market, p_model, y_all)
    pool_samples = _bootstrap_rows(_block_ids(frame), min(n_boot, 500), rng)
    boot_beta = np.array([fit_beta(p_market[r], p_model[r], y_all[r]) for r in pool_samples])

    by_season = {int(s): float(d[(f_ev.season == s).to_numpy()].mean()) for s in eval_seasons}
    by_league = {
        str(lg): float(d[(f_ev.league == lg).to_numpy()].mean())
        for lg in sorted(f_ev.league.unique())
    }
    res = StackResult(
        eval_seasons=eval_seasons,
        beta_by_season=betas,
        delta=float(d.mean()),
        delta_ci=(float(np.quantile(boot_delta, lo_q)), float(np.quantile(boot_delta, hi_q))),
        beta_pooled=beta_pooled,
        beta_ci=(float(np.quantile(boot_beta, lo_q)), float(np.quantile(boot_beta, hi_q))),
        brier_market=brier_score(pm, y),
        brier_blend=brier_score(pq, y),
        delta_by_season=by_season,
        delta_by_league=by_league,
        log_loss_market=log_loss(pm, y),
        log_loss_blend=log_loss(pq, y),
        log_loss_model=log_loss(pmod, y),
        n=int(ev.sum()),
        level=level,
    )
    res.criteria = {
        "improvement >= 0.005": res.delta >= 0.005,
        "improvement interval above 0": res.delta_ci[0] > 0,
        "beta interval above 0": res.beta_ci[0] > 0,
        "Brier not worse (+0.001)": res.brier_blend <= res.brier_market + 0.001,
        "positive in >= 4 of 6 seasons": sum(v > 0 for v in by_season.values()) >= 4,
        "positive in every league": all(v > 0 for v in by_league.values()),
    }
    return res


def render_report(res: StackResult, model: str, market: str = "1x2") -> str:
    tick = {True: "PASS", False: "FAIL"}
    lines = [
        f"# Stack test: {model}, market {market}",
        "",
        f"**Verdict: {'PASS' if res.passed else 'FAIL'}**  (all criteria must pass; "
        "pre-registered in docs/adr/0003 and 0005)",
        "",
        f"Evaluation: seasons {res.eval_seasons[0]}-{res.eval_seasons[-1]}, {res.n} matches.",
        "",
        "| Forecast | Log-loss |",
        "|---|---|",
        f"| Model alone | {res.log_loss_model:.4f} |",
        f"| Market alone (closing, margin removed) | {res.log_loss_market:.4f} |",
        f"| Blend (market + beta x model) | {res.log_loss_blend:.4f} |",
        "",
        f"Improvement of blend over market: {res.delta:+.5f} per match "
        f"({res.level:.2%} interval {res.delta_ci[0]:+.5f} to {res.delta_ci[1]:+.5f})",
        f"Pooled beta: {res.beta_pooled:.3f} "
        f"({res.level:.2%} interval {res.beta_ci[0]:.3f} to {res.beta_ci[1]:.3f})",
        f"Brier: market {res.brier_market:.4f}, blend {res.brier_blend:.4f}",
        "",
        "| Criterion | Result |",
        "|---|---|",
        *[f"| {k} | {tick[v]} |" for k, v in res.criteria.items()],
        "",
        "| Season | beta used (fitted on earlier seasons) | Improvement |",
        "|---|---|---|",
        *[
            f"| {s} | {res.beta_by_season[s]:.3f} | {res.delta_by_season[s]:+.5f} |"
            for s in res.eval_seasons
        ],
        "",
        "| League | Improvement |",
        "|---|---|",
        *[f"| {lg} | {v:+.5f} |" for lg, v in res.delta_by_league.items()],
        "",
    ]
    return "\n".join(lines)
