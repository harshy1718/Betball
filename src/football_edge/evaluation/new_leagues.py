"""Pre-registered check for adding leagues (docs/adr/0024): does the model work in the new
leagues, and does fitting them jointly leave the old leagues as good as before?"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.evaluation.challenger import match_losses
from football_edge.evaluation.folds import make_folds
from football_edge.evaluation.runner import _RESULT_COLS
from football_edge.leagues import name
from football_edge.models.calibration import block_bootstrap
from football_edge.models.dixon_coles import DixonColes, DixonColesParams


def _predict(model: DixonColes, test: pd.DataFrame) -> pd.DataFrame:
    hidden = test.drop(columns=_RESULT_COLS, errors="ignore")
    p = model.predict_proba(hidden)
    extra = model.predict_extra(hidden)
    return pd.DataFrame(
        {
            "p_h": p[:, 0],
            "p_d": p[:, 1],
            "p_a": p[:, 2],
            "p_over25": extra["p_over25"],
            "p_btts": extra["p_btts"],
        }
    )


def walk(
    matches: pd.DataFrame, params: dict[str, Any], old: list[str], lo: int, hi: int, min_train: int
) -> pd.DataFrame:
    """Per test match: the all-league fit's chances and, for old leagues, the old-only fit's."""
    rows = []
    for f in make_folds(matches, lo, hi, min_train):
        train, test = matches.iloc[f.train_idx], matches.iloc[f.test_idx]
        everything = DixonColes(DixonColesParams.from_dict(params))
        everything.fit(train)
        a = _predict(everything, test).add_suffix("_all")
        out = pd.concat(
            [test[["league", "season", "date", "FTHG", "FTAG"]].reset_index(drop=True), a], axis=1
        )
        out["week"] = str(f.snapshot.date())
        is_old = test["league"].isin(old).to_numpy()
        for c in ("p_h", "p_d", "p_a", "p_over25", "p_btts"):
            out[f"{c}_old"] = np.nan
        if is_old.any():
            only = DixonColes(DixonColesParams.from_dict(params))
            only.fit(train[train["league"].isin(old)])
            b = _predict(only, test[is_old])
            for c in b.columns:
                out.loc[is_old, f"{c}_old"] = b[c].to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def baseline(matches: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Each league's rates over the 3 seasons before the match's season."""
    m = matches.dropna(subset=["FTHG", "FTAG"])
    rates = {}
    for (lg, s), _ in rows.groupby(["league", "season"]):
        yr = int(str(s))
        h = m[(m["league"] == lg) & m["season"].between(yr - 3, yr - 1)]
        hg, ag = h["FTHG"], h["FTAG"]
        rates[(lg, s)] = (
            (hg > ag).mean(),
            (hg == ag).mean(),
            (hg < ag).mean(),
            (hg + ag >= 3).mean(),
            ((hg > 0) & (ag > 0)).mean(),
        )
    r = np.array([rates[(lg, s)] for lg, s in zip(rows["league"], rows["season"], strict=True)])
    return pd.DataFrame(r, columns=["p_h", "p_d", "p_a", "p_over25", "p_btts"])


def _losses(rows: pd.DataFrame, cols: pd.DataFrame) -> pd.DataFrame:
    return match_losses(
        cols.assign(goals_home=rows["FTHG"].to_numpy(int), goals_away=rows["FTAG"].to_numpy(int))
    )


def check(
    matches: pd.DataFrame, params: dict[str, Any], old: list[str], new: list[str], min_train: int
) -> tuple[str, bool]:
    cols = ["p_h", "p_d", "p_a", "p_over25", "p_btts"]
    lines = [
        "# Adding " + " and ".join(name(x) for x in new) + " (ADR 0024)",
        "",
        "Log loss per match (result + over/under 2.5 + both-score), lower is better.",
        "",
    ]
    ok = True
    for lo, hi, label in ((2020, 2025, "2020/21-2025/26"), (2026, 2026, "2026/27 so far")):
        r = walk(matches, params, old, lo, hi, min_train)
        all_l = _losses(r, r[[f"{c}_all" for c in cols]].set_axis(cols, axis=1))
        base = _losses(r, baseline(matches, r))
        lines += [
            f"## {label}",
            "",
            "| League | Matches | Nox (5-league fit) | Nox (old fit) | League baseline |",
            "|---|---|---|---|---|",
        ]
        for lg in [*old, *new]:
            s = (r["league"] == lg).to_numpy()
            if not s.any():
                continue
            old_txt = "-"
            if lg in old:
                o = _losses(r[s], r.loc[s, [f"{c}_old" for c in cols]].set_axis(cols, axis=1))
                old_txt = f"{o['sum'].mean():.4f}"
            lines.append(
                f"| {name(lg)} | {s.sum():,} | {all_l['sum'][s].mean():.4f} | {old_txt} | "
                f"{base['sum'][s].mean():.4f} |"
            )
        so = r["league"].isin(old).to_numpy()
        o = _losses(r[so], r.loc[so, [f"{c}_old" for c in cols]].set_axis(cols, axis=1))
        d_old = float(all_l["sum"][so].mean() - o["sum"].mean())
        lines += [
            "",
            f"* Old leagues: 5-league fit minus old fit {d_old:+.5f} per match "
            f"(rule: not worse than +0.00100) -> {'PASS' if d_old <= 0.001 else 'FAIL'}",
        ]
        ok &= d_old <= 0.001
        for lg in new:
            s = (r["league"] == lg).to_numpy()
            d = (all_l["sum"][s] - base["sum"][s]).to_numpy()
            if lo == 2020:
                a, b = block_bootstrap(d, r.loc[s, "week"].to_numpy())
                good = b < 0
                lines.append(
                    f"* {name(lg)}: Nox minus baseline {d.mean():+.5f} per match (95% "
                    f"interval {a:+.5f} to {b:+.5f}) -> {'PASS' if good else 'FAIL'}"
                )
            else:
                good = bool(d.mean() < 0)
                lines.append(
                    f"* {name(lg)}: Nox minus baseline {d.mean():+.5f} per match -> "
                    f"{'PASS' if good else 'FAIL'}"
                )
            ok &= bool(good)
        lines.append("")
    lines.append(f"**{'PASS: add the leagues' if ok else 'FAIL: the leagues stay out'}.**")
    return "\n".join(lines), ok
