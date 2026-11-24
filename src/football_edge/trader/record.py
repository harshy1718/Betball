"""Track record for the extra goal bets (team goals, other goal lines, handicaps, win to nil,
odd/even), so every desk leg gets a real confidence label instead of a guess.

Built from the same walk-forward predictions as reports/confidence/track_record.csv (each
match predicted only from matches before it, seasons 2020/21-2025/26): the saved expected goals
are turned into a score grid and every bet is read from it, then compared with what happened.
Same bands and the same High / Medium / Low rules as predict/confidence.py.

The grid needs the Dixon-Coles rho, which the predictions file does not keep; the rho of the
current fit is used. It only moves the 0-0, 1-0, 0-1 and 1-1 cells slightly; the check in
`compare_ou25` shows the effect on over/under 2.5 against the saved probabilities.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from football_edge.models.calibration import adjust_draws
from football_edge.models.dixon_coles import score_grid, score_grids
from football_edge.predict.confidence import BAND, HIGH_GAP, HIGH_N, LOW_GAP, LOW_N, _skill

MAX_GOALS = 10

# market key -> list of (selection name, test on (home goals, away goals))
Test = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _cmp(x: np.ndarray, line: float, over: bool) -> np.ndarray:
    out: np.ndarray = x > line if over else x < line
    return out


def _total(line: float, over: bool) -> Test:
    return lambda h, a: _cmp(h + a, line, over)


def _team(home: bool, line: float, over: bool) -> Test:
    return lambda h, a: _cmp(h if home else a, line, over)


def _margin(home: bool, k: int, by: bool) -> Test:
    """by=True: that side wins by k+; by=False: it does not (the other side's +handicap)."""
    return lambda h, a: (
        ((h - a) if home else (a - h)) >= k if by else ((h - a) if home else (a - h)) < k
    )


def _wtn(home: bool) -> Test:
    return lambda h, a: (h > 0) & (a == 0) if home else (a > 0) & (h == 0)


def _odd(odd: bool) -> Test:
    return lambda h, a: (h + a) % 2 == (1 if odd else 0)


def goal_markets() -> dict[str, list[tuple[str, Test]]]:
    out: dict[str, list[tuple[str, Test]]] = {}
    for line in (0.5, 1.5, 3.5, 4.5):
        out[f"goals_{line}"] = [("over", _total(line, True)), ("under", _total(line, False))]
    for side in ("home", "away"):
        for line in (0.5, 1.5, 2.5):
            home = side == "home"
            out[f"{side}_goals_{line}"] = [
                ("over", _team(home, line, True)),
                ("under", _team(home, line, False)),
            ]
    for k in (2, 3):  # handicap 1.5 and 2.5: "home -1.5" = home by 2+, its other side "away +1.5"
        out[f"hcp_home{k}"] = [
            (f"home-{k - 0.5:g}", _margin(True, k, True)),
            (f"away+{k - 0.5:g}", _margin(True, k, False)),
        ]
        out[f"hcp_away{k}"] = [
            (f"away-{k - 0.5:g}", _margin(False, k, True)),
            (f"home+{k - 0.5:g}", _margin(False, k, False)),
        ]
    out["wtn"] = [("home", _wtn(True)), ("away", _wtn(False))]
    out["oddeven"] = [("odd", _odd(True)), ("even", _odd(False))]
    return out


def grid(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """The Dixon-Coles score grid (the same function DixonColes.score_matrix uses)."""
    return score_grid(lam, mu, rho, max_goals)


def build_goal_record(
    preds: pd.DataFrame, rho: float, seasons: tuple[int, int] = (2020, 2025)
) -> pd.DataFrame:
    """Per (market, band): n, mean predicted, hit rate, skill and label, as track_record.csv."""
    p = preds[preds["season"].between(*seasons)].dropna(subset=["exp_home", "exp_away"])
    h_obs, a_obs = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    gh, ga = np.indices((MAX_GOALS + 1, MAX_GOALS + 1))
    markets = goal_markets()
    grids = score_grids(p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float), rho)
    if "cal_delta" in p.columns:  # calibrated predictions: each row's own draw adjustment
        grids = adjust_draws(grids, p["cal_delta"].to_numpy(float))
    probs = {
        (m, s): (grids * t(gh, ga)[None]).sum(axis=(1, 2))
        for m, sels in markets.items()
        for s, t in sels
    }
    parts = []
    for m, sels in markets.items():
        for s, t in sels:
            parts.append(pd.DataFrame({"market": m, "prob": probs[(m, s)], "hit": t(h_obs, a_obs)}))
    rows = pd.concat(parts, ignore_index=True)
    rows["band"] = (np.floor(rows["prob"] / BAND) * BAND).round(2)
    skill = {
        m: _skill(r["prob"].to_numpy(), r["hit"].to_numpy()) for m, r in rows.groupby("market")
    }
    table = (
        rows.groupby(["market", "band"])
        .agg(n=("hit", "size"), predicted=("prob", "mean"), hit_rate=("hit", "mean"))
        .reset_index()
    )
    table.insert(0, "source", "model")
    table["skill"] = table["market"].map(skill)
    gap = (table["hit_rate"] - table["predicted"]).abs()
    table["label"] = np.where(
        ~table["skill"] | (table["n"] < LOW_N) | (gap > LOW_GAP),
        "Low",
        np.where((table["n"] >= HIGH_N) & (gap <= HIGH_GAP), "High", "Medium"),
    )
    return table


def compare_ou25(preds: pd.DataFrame, rho: float) -> float:
    """Largest gap between the saved over-2.5 chance and the one rebuilt here (a sanity check)."""
    p = preds.dropna(subset=["exp_home", "exp_away", "p_over25"]).head(500)
    gh, ga = np.indices((MAX_GOALS + 1, MAX_GOALS + 1))
    rebuilt = [
        float(grid(lam, mu, rho)[gh + ga >= 3].sum())
        for lam, mu in zip(p["exp_home"], p["exp_away"], strict=True)
    ]
    return float(np.max(np.abs(np.array(rebuilt) - p["p_over25"].to_numpy(float))))
