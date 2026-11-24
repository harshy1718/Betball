"""Pre-registered test of a referee effect on Premier League cards (docs/adr/0021).

Two cards models, identical except for the referee effect, both trained on Premier League
matches only, weekly walk-forward. Score: summed log loss of over 3.5-6.5 cards per match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.evaluation.folds import make_folds
from football_edge.models.calibration import CARD_LINES, block_bootstrap
from football_edge.models.counts import CountParams, TeamCountModel, with_card_totals

EPS = 1e-12


def _losses(model: TeamCountModel, test: pd.DataFrame, total: np.ndarray) -> np.ndarray:
    out = np.zeros(len(test))
    for x in CARD_LINES:
        q = np.clip(model.p_over(test, x), EPS, 1 - EPS)
        out -= np.where(total > x, np.log(q), np.log(1 - q))
    return out


def walk_forward(matches: pd.DataFrame, lo: int, hi: int, min_train: int = 1500) -> pd.DataFrame:
    e0 = with_card_totals(matches[matches["league"] == "E0"].reset_index(drop=True))
    e0 = e0.dropna(subset=["HCARDS", "ACARDS"]).reset_index(drop=True)
    rows = []
    for fold in make_folds(e0, lo, hi, min_train):
        train, test = e0.iloc[fold.train_idx], e0.iloc[fold.test_idx]
        total = (test["HCARDS"] + test["ACARDS"]).to_numpy(float)
        hidden = test.drop(columns=["HY", "AY", "HR", "AR", "HCARDS", "ACARDS"])
        base = TeamCountModel("cards")
        ref = TeamCountModel("cards", CountParams(referee_effects=True))
        base.fit(train)
        ref.fit(train)
        rows.append(
            pd.DataFrame(
                {
                    "season": test["season"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "referee": test["referee"].to_numpy(),
                    "week": str(fold.snapshot.date()),
                    "loss_base": _losses(base, hidden, total),
                    "loss_ref": _losses(ref, hidden, total),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def run(matches: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    main = walk_forward(matches, 2020, 2025)
    conf = walk_forward(matches, 2026, 2026)
    d = (main["loss_base"] - main["loss_ref"]).to_numpy()
    lo, hi = block_bootstrap(d, main["week"].to_numpy())
    d26 = (conf["loss_base"] - conf["loss_ref"]).to_numpy()
    lo26, hi26 = block_bootstrap(d26, conf["week"].to_numpy()) if len(conf) else (np.nan, np.nan)
    passed = lo > 0 and len(conf) > 0 and d26.mean() > 0
    by = (
        main.assign(d=d)
        .groupby("season")
        .agg(
            n=("d", "size"),
            base=("loss_base", "mean"),
            ref=("loss_ref", "mean"),
            gain=("d", "mean"),
        )
    )
    text = "\n".join(
        [
            "# Referee effect on Premier League cards (ADR 0021)",
            "",
            "Summed log loss of over 3.5/4.5/5.5/6.5 cards per match, lower is better.",
            "",
            by.round(4).to_string(),
            "",
            f"2020/21-2025/26: without referee {main['loss_base'].mean():.5f}, with referee "
            f"{main['loss_ref'].mean():.5f}; improvement {d.mean():+.5f} per match "
            f"(95% interval {lo:+.5f} to {hi:+.5f}; {len(main):,} matches).",
            f"2026/27 so far: without {conf['loss_base'].mean():.5f}, "
            f"with {conf['loss_ref'].mean():.5f}; improvement {d26.mean():+.5f} "
            f"(95% interval {lo26:+.5f} to {hi26:+.5f}; {len(conf)} matches).",
            "",
            f"**{'PASS: adopt the referee effect' if passed else 'FAIL: the referee stays out'}.**",
        ]
    )
    return text, pd.concat([main, conf], ignore_index=True)
