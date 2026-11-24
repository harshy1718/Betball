"""Champion vs challenger on the same folds (ADR 0022): both models refitted with today's code
on every weekly fold, scored per match on match result + over/under 2.5 + both-score."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.evaluation.folds import make_folds
from football_edge.evaluation.runner import walk_forward
from football_edge.models.calibration import block_bootstrap
from football_edge.models.registry import model_factory

EPS = 1e-12


def match_losses(p: pd.DataFrame) -> pd.DataFrame:
    """Per match log loss of match result, over/under 2.5 and both-score, and their sum."""
    h, a = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    res = np.where(h > a, 0, np.where(h == a, 1, 2))
    p3 = p[["p_h", "p_d", "p_a"]].to_numpy(float)[np.arange(len(p)), res]
    ov = np.clip(p["p_over25"].to_numpy(float), EPS, 1 - EPS)
    bt = np.clip(p["p_btts"].to_numpy(float), EPS, 1 - EPS)
    out = pd.DataFrame(
        {
            "1x2": -np.log(np.clip(p3, EPS, 1)),
            "ou25": -np.where(h + a >= 3, np.log(ov), np.log(1 - ov)),
            "btts": -np.where((h > 0) & (a > 0), np.log(bt), np.log(1 - bt)),
        }
    )
    out["sum"] = out.sum(axis=1)
    return out


def run_models(
    matches: pd.DataFrame, models: dict[str, dict[str, Any]], lo: int, hi: int, min_train: int
) -> dict[str, pd.DataFrame]:
    folds = make_folds(matches, lo, hi, min_train)
    return {
        name: walk_forward(matches, folds, model_factory(name, params))
        for name, params in models.items()
    }


def compare(champ: pd.DataFrame, chall: pd.DataFrame) -> dict[str, Any]:
    """Paired comparison; negative difference = the challenger is better."""
    a, b = match_losses(champ), match_losses(chall)
    d = (b["sum"] - a["sum"]).to_numpy()
    snap = pd.to_datetime(champ["snapshot"]).dt.strftime("%Y-%m-%d")
    lo, hi = block_bootstrap(d, (champ["league"].astype(str) + "|" + snap).to_numpy())
    return {
        "n": len(d),
        "champion": {k: float(a[k].mean()) for k in a.columns},
        "challenger": {k: float(b[k].mean()) for k in b.columns},
        "diff": float(d.mean()),
        "lo": lo,
        "hi": hi,
        "by_season": pd.DataFrame(
            {"season": champ["season"].to_numpy(), "champion": a["sum"], "challenger": b["sum"]}
        )
        .groupby("season")
        .mean(),
    }


def verdict(main: dict[str, Any], conf: dict[str, Any], champ: str, chall: str) -> tuple[str, bool]:
    passed = main["hi"] < 0 and conf["diff"] < 0
    outcome = (
        "PASS: the challenger replaces the locked model"
        if passed
        else "FAIL: the locked model stays"
    )
    lines = [
        f"# {chall} against {champ} (ADR 0022)",
        "",
        "Log loss per match, lower is better. Both models refitted on every weekly fold.",
        "",
        "| | " + champ + " | " + chall + " |",
        "|---|---|---|",
    ]
    for k, name in (
        ("1x2", "Match result"),
        ("ou25", "Over/under 2.5"),
        ("btts", "Both score"),
        ("sum", "Sum of the three"),
    ):
        lines.append(f"| {name} | {main['champion'][k]:.5f} | {main['challenger'][k]:.5f} |")
    lines += [
        "",
        "By season (sum of the three):",
        "",
        main["by_season"].round(4).to_string(),
        "",
        f"2020/21-2025/26: difference {main['diff']:+.5f} per match (95% interval "
        f"{main['lo']:+.5f} to {main['hi']:+.5f}; {main['n']:,} matches).",
        f"2026/27 so far: {conf['champion']['sum']:.5f} -> {conf['challenger']['sum']:.5f}, "
        f"difference {conf['diff']:+.5f} (95% interval {conf['lo']:+.5f} to "
        f"{conf['hi']:+.5f}; {conf['n']} matches).",
        "",
        f"**{outcome}.**",
    ]
    return "\n".join(lines), passed
