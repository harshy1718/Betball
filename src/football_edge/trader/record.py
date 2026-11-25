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
from typing import Any

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
    return _table(pd.concat(parts, ignore_index=True))


def _table(rows: pd.DataFrame) -> pd.DataFrame:
    """rows (market, prob, hit) -> per (market, band) n, predicted, hit rate, skill, label."""
    rows = rows.copy()
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


def new_market_rows(
    preds: pd.DataFrame, rho: float, matches: pd.DataFrame, seasons: tuple[int, int] = (2020, 2025)
) -> pd.DataFrame:
    """One row per (match, ADR 0028 market): the model's chance and whether it came in.
    Refunded bets and matches without a half-time score (half markets) are left out. Half
    shares are walk-forward: each season uses only the seasons before it."""
    from football_edge.trader.markets import CATALOGUE, chances, half_shares

    p = preds[preds["season"].between(*seasons)].dropna(subset=["exp_home", "exp_away"])
    p = p.reset_index(drop=True)
    m = matches.reset_index(drop=True)
    ht = m.loc[p["row"], ["HTHG", "HTAG"]].to_numpy(float)
    grids = score_grids(p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float), rho)
    if "cal_delta" in p.columns:
        grids = adjust_draws(grids, p["cal_delta"].to_numpy(float))
    sh, sa = np.full(len(p), np.nan), np.full(len(p), np.nan)
    for s in p["season"].unique():
        shares = half_shares(m, before_season=int(s))
        idx = (p["season"] == s).to_numpy()
        sh[idx] = p.loc[idx, "league"].map({k: v[0] for k, v in shares.items()})
        sa[idx] = p.loc[idx, "league"].map({k: v[1] for k, v in shares.items()})
    probs = chances(grids, CATALOGUE, (sh, sa))
    h, a = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    has_ht = np.isfinite(ht).all(axis=1)
    h1, a1 = np.nan_to_num(ht[:, 0]).astype(int), np.nan_to_num(ht[:, 1]).astype(int)
    parts = []
    for j, mk in enumerate(CATALOGUE):
        win = np.asarray(mk.win(h, a, h1, a1), dtype=bool)
        keep = np.isfinite(probs[:, j]) & (has_ht if mk.half else True)
        if mk.void is not None:
            keep &= ~np.asarray(mk.void(h, a, h1, a1), dtype=bool)
        parts.append(pd.DataFrame({
            "market": mk.key, "leg_market": mk.market, "sel": mk.sel, "line": mk.line,
            "row": p.loc[keep, "row"].to_numpy(), "season": p.loc[keep, "season"].to_numpy(),
            "league": p.loc[keep, "league"].to_numpy(), "prob": probs[keep, j], "hit": win[keep],
        }))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def build_new_record(rows: pd.DataFrame) -> pd.DataFrame:
    """Track record (same columns as build_goal_record) for the ADR 0028 markets."""
    return _table(rows[["market", "prob", "hit"]])


def new_market_summary(rows: pd.DataFrame) -> str:
    """Markdown report (ADR 0028): per selection, how often it came in against the model's
    average chance, and log loss against always predicting the overall rate."""
    from football_edge.trader.markets import CATALOGUE

    labels = {(m.market, m.sel, m.line): m.label.format(h="Home", a="Away") for m in CATALOGUE}

    def ll(p: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))

    lines = ["# New markets (ADR 0028): walk-forward check, 2020/21-2025/26", "",
             "Each match predicted only from earlier matches. 'Gain' = how much lower the log "
             "loss is than always predicting the overall rate (above 0 = skill). Refunded bets "
             "are left out of the counts.", "",
             "| Market | Matches | Model's average chance | Came in | Gain vs average | Skill |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for (lm, sel, line), g in rows.groupby(["leg_market", "sel", "line"], sort=False, dropna=False):
        ln = float(line) if isinstance(line, int | float) and line == line else None  # NaN -> None
        key = (str(lm), str(sel), ln)
        p, y = g["prob"].to_numpy(float), g["hit"].to_numpy(bool)
        gain = 1 - ll(p, y) / ll(np.full(len(g), y.mean()), y)
        lines.append(f"| {labels.get(key, f'{lm} {sel}')} | {len(g):,} | {p.mean():.1%} | "
                     f"{y.mean():.1%} | {gain:+.1%} | {'yes' if gain > 0 else 'no'} |")  # fmt: skip
    return "\n".join(lines) + "\n"


BEST_GAIN, BEST_GAP, BEST_N = 0.05, 0.015, 1000  # ADR 0029


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(y, np.log(q), np.log(1 - q))))


def _skill_row(market: str, sel: str, line: float | None, p: np.ndarray, y: np.ndarray) -> dict:  # type: ignore[type-arg]
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    gain = 1 - _ll(p, y) / _ll(np.full(len(y), y.mean()), y) if len(y) else float("nan")
    return {"market": market, "sel": sel, "line": line, "n": len(y), "predicted": float(p.mean()),
            "hit_rate": float(y.mean()), "gain": gain}  # fmt: skip


def market_skill(
    preds: pd.DataFrame,
    counts: pd.DataFrame | None,
    matches: pd.DataFrame,
    rho: float,
    seasons: tuple[int, int] = (2020, 2025),
) -> pd.DataFrame:
    """ADR 0029: per selection (market, sel, line) of every desk bet, walk-forward n, average
    chance, hit rate, gain in log loss over always predicting the overall rate, and `best`."""
    from football_edge.trader.legs import LegFactory, goal_test

    p = preds[preds["season"].between(*seasons)].dropna(subset=["exp_home", "exp_away"])
    grids = score_grids(p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float), rho)
    if "cal_delta" in p.columns:
        grids = adjust_draws(grids, p["cal_delta"].to_numpy(float))
    h, a = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    gh, ga = np.indices(grids.shape[1:])
    specs: list[tuple[str, str, float | None]] = [
        ("1x2", "home", None), ("1x2", "draw", None), ("1x2", "away", None),
        ("dc", "1X", None), ("dc", "X2", None), ("dc", "12", None),
        ("ou25", "over", 2.5), ("ou25", "under", 2.5), ("btts", "yes", None), ("btts", "no", None),
    ]  # fmt: skip
    specs += [(mk, s, ln) for mk, s, ln, _ in LegFactory._extra_bets("H", "A")]
    rows = []
    for mk, s, ln in specs:
        t = goal_test(mk, s, ln)
        prob = (grids * t(gh, ga)[None]).sum(axis=(1, 2))
        rows.append(_skill_row(mk, s, ln, prob, t(h, a)))
    new = new_market_rows(preds, rho, matches, seasons)
    for (lm, sel, line), g in new.groupby(["leg_market", "sel", "line"], sort=False, dropna=False):
        ln = float(line) if isinstance(line, int | float) and line == line else None
        rows.append(_skill_row(str(lm), str(sel), ln, g["prob"].to_numpy(), g["hit"].to_numpy()))
    if counts is not None:
        c = counts[(counts["stat"] == "cards") & counts["season"].between(*seasons)]
        for line in (3.5, 4.5, 5.5):
            col = f"p_team_{line}"
            if col not in c.columns:
                continue
            k = c.dropna(subset=[col, "total"])
            pr, tot = k[col].to_numpy(float), k["total"].to_numpy(float)
            rows.append(_skill_row("cards", "over", line, pr, tot > line))
            rows.append(_skill_row("cards", "under", line, 1 - pr, tot < line))
    out = pd.DataFrame(rows)
    out["best"] = (
        (out["gain"] >= BEST_GAIN)
        & ((out["predicted"] - out["hit_rate"]).abs() <= BEST_GAP)
        & (out["n"] >= BEST_N)
    )
    return out


def best_set(skill: pd.DataFrame) -> set[tuple[str, str, float | None]]:
    """{(market, sel, line)} of the best markets in a market_skill table."""
    out = set()
    for r in skill[skill["best"].astype(bool)].itertuples(index=False):
        out.add((str(r.market), str(r.sel), _line(r.line)))
    return out


def best_markets_summary(skill: pd.DataFrame) -> str:
    """Markdown: every selection's gain, best ones first (ADR 0029)."""
    names = bet_names()
    t = skill.sort_values(["best", "gain"], ascending=[False, False])
    lines = ["# Best markets (ADR 0029)", "",
             f"Best = log loss at least {BEST_GAIN:.0%} lower than always predicting the overall "
             f"rate, average chance within {BEST_GAP * 100:.1f} points of the hit rate, "
             f"{BEST_N:,}+ matches. Walk-forward, 2020/21-2025/26.", "",
             "| Bet | Line | Matches | Model's average | Came in | Gain | Best |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in t.itertuples(index=False):
        line = _line(r.line)
        name = names.get((str(r.market), str(r.sel), line), f"{r.market} {r.sel}")
        lines.append(f"| {name} | {'' if line is None else f'{line:g}'} | {r.n:,} | "
                     f"{r.predicted:.1%} | {r.hit_rate:.1%} | {r.gain:+.1%} | "
                     f"{'yes' if r.best else ''} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def _line(v: Any) -> float | None:
    """A line read from a table: None for blank / NaN."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def bet_names() -> dict[tuple[str, str, float | None], str]:
    """(market, sel, line) -> a readable name, for every desk bet (Home / Away for the teams)."""
    from football_edge.trader.legs import LegFactory
    from football_edge.trader.markets import CATALOGUE

    out: dict[tuple[str, str, float | None], str] = {
        ("1x2", "home", None): "Home win", ("1x2", "draw", None): "Draw",
        ("1x2", "away", None): "Away win", ("dc", "1X", None): "Home or draw",
        ("dc", "X2", None): "Away or draw", ("dc", "12", None): "Home or away (no draw)",
        ("ou25", "over", 2.5): "Over 2.5 goals", ("ou25", "under", 2.5): "Under 2.5 goals",
        ("btts", "yes", None): "Both teams score", ("btts", "no", None): "Not both teams score",
    }  # fmt: skip
    for mk, sel, ln, label in LegFactory._extra_bets("Home", "Away"):
        out[(mk, sel, ln)] = label
    for m in CATALOGUE:
        out[(m.market, m.sel, m.line)] = m.label.format(h="Home", a="Away")
    for ln in (3.5, 4.5, 5.5):
        out[("cards", "over", ln)] = f"Over {ln:g} cards"
        out[("cards", "under", ln)] = f"Under {ln:g} cards"
    return out
