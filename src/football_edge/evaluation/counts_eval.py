"""Walk-forward evaluation of the corners and cards models against league averages.

Same weekly folds as the goals models: fit on matches before Monday, predict that week.
For every standard line we score P(over) with log-loss, for the team model and for the
league-average baseline (same distribution, no team effects). There are no bookmaker
prices for these markets in the data, so this measures accuracy only, not value.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from football_edge.evaluation.folds import Fold
from football_edge.evaluation.metrics import log_loss
from football_edge.models.counts import STATS, CountParams, TeamCountModel, with_card_totals

log = logging.getLogger(__name__)


def walk_forward_counts(matches: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """One row per test match and stat: totals and P(over line) for team and baseline."""
    parts = []
    matches = with_card_totals(matches)
    for n, fold in enumerate(folds, start=1):
        train, test = matches.iloc[fold.train_idx], matches.iloc[fold.test_idx]
        hidden = test.drop(columns=[c for s in STATS.values() for c in s[:2]], errors="ignore")
        for stat, (hcol, acol, lines) in STATS.items():
            models = {
                "team": TeamCountModel(stat),
                "base": TeamCountModel(stat, CountParams(team_effects=False)),
            }
            out = pd.DataFrame(
                {
                    "row": fold.test_idx,
                    "season": test["season"].to_numpy(),
                    "league": test["league"].to_numpy(),
                    "stat": stat,
                    "total": (test[hcol] + test[acol]).to_numpy(),
                }
            )
            for name, m in models.items():
                m.fit(train)
                lam, mu = m.expected(hidden)
                out[f"exp_{name}"] = lam + mu
                for line in lines:
                    out[f"p_{name}_{line}"] = m.p_over(hidden, line)
            parts.append(out)
        if n % 50 == 0:
            log.info("counts: %d/%d folds", n, len(folds))
    return pd.concat(parts, ignore_index=True)


def summarise_counts(preds: pd.DataFrame) -> pd.DataFrame:
    """Log-loss per stat and line: team model, baseline, and the improvement."""
    rows = []
    for stat, (_, _, lines) in STATS.items():
        g = preds[(preds["stat"] == stat) & preds["total"].notna()]
        for line in lines:
            y = np.asarray(g["total"] > line, dtype=np.intp)
            scores = {}
            for name in ("team", "base"):
                p = g[f"p_{name}_{line}"].to_numpy().clip(1e-6, 1 - 1e-6)
                scores[name] = log_loss(np.c_[1 - p, p], y)
            rows.append(
                {
                    "stat": stat,
                    "line": f"over {line}",
                    "n": len(g),
                    "actual_rate": float(y.mean()),
                    "avg_prob_team": float(g[f"p_team_{line}"].mean()),
                    "log_loss_team": scores["team"],
                    "log_loss_base": scores["base"],
                    "improvement": scores["base"] - scores["team"],
                }
            )
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame, stat: str, line: float, bins: int = 5) -> pd.DataFrame:
    """Predicted vs actual frequency, in bins of the team model's probability."""
    g = preds[(preds["stat"] == stat) & preds["total"].notna()].copy()
    g["bin"] = pd.qcut(g[f"p_team_{line}"], bins, duplicates="drop")
    return g.groupby("bin", observed=True).agg(
        predicted=(f"p_team_{line}", "mean"),
        actual=("total", lambda t: float((t > line).mean())),
        n=("total", "size"),
    )
