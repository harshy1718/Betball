"""Confidence: how far to trust each probability, from its own track record.

A probability says how likely an outcome is. Confidence says how reliable that number has
been. From the walk-forward predictions of 2020/21-2025/26 (never trained on the week
they predicted) we build a track record per source (model or market), market and
probability band of 5 points: how many such predictions there were and how often they
came true.

Label rule, fixed before looking at any result:
  * Low    - the market shows no skill over a plain base rate (log-loss not better), or
             fewer than 200 past predictions in this band, or the band's hit rate is more
             than 6 points away from what was predicted.
  * High   - skill over the base rate, 500+ past predictions in the band, and the hit
             rate within 3 points of the prediction.
  * Medium - everything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_edge.evaluation.metrics import log_loss

BAND = 0.05
LOW_N, HIGH_N = 200, 500
HIGH_GAP, LOW_GAP = 0.03, 0.06


def _rows(source: str, market: str, prob: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"source": source, "market": market, "prob": prob, "hit": hit})


def _skill(prob: np.ndarray, hit: np.ndarray) -> bool:
    """Better log-loss than always predicting the overall frequency (two-way view)."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    y = np.asarray(hit, dtype=np.intp)
    base = np.full_like(p, y.mean())
    return log_loss(np.c_[1 - p, p], y) < log_loss(np.c_[1 - base, base], y)


def build_track_record(
    goals_preds: pd.DataFrame,
    counts_preds: pd.DataFrame | None,
    matches: pd.DataFrame,
    seasons: tuple[int, int] = (2020, 2025),
) -> pd.DataFrame:
    """Per (source, market, band): n, mean predicted, hit rate, skill flag and label."""
    g = goals_preds[goals_preds["season"].between(*seasons)].reset_index(drop=True)
    d = (g["goals_home"] - g["goals_away"]).to_numpy()
    tot = (g["goals_home"] + g["goals_away"]).to_numpy()
    btts = ((g["goals_home"] > 0) & (g["goals_away"] > 0)).to_numpy()
    parts = [
        _rows("model", "1x2", g["p_h"].to_numpy(), d > 0),
        _rows("model", "1x2", g["p_d"].to_numpy(), d == 0),
        _rows("model", "1x2", g["p_a"].to_numpy(), d < 0),
        _rows("model", "ou25", g["p_over25"].to_numpy(), tot >= 3),
        _rows("model", "ou25", 1 - g["p_over25"].to_numpy(), tot < 3),
        _rows("model", "btts", g["p_btts"].to_numpy(), btts),
        _rows("model", "btts", 1 - g["p_btts"].to_numpy(), ~btts),
    ]
    # The market's own fair probabilities (closing prices, margin removed).
    mk = matches.iloc[g["row"].to_numpy()].reset_index(drop=True)
    o = mk[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
    ok = ~np.isnan(o).any(axis=1)
    fair = (1 / o[ok]) / (1 / o[ok]).sum(axis=1, keepdims=True)
    for k, won in enumerate((d[ok] > 0, d[ok] == 0, d[ok] < 0)):
        parts.append(_rows("market", "1x2", fair[:, k], won))
    o = mk[["ou_close_over", "ou_close_under"]].to_numpy(dtype=float)
    ok = ~np.isnan(o).any(axis=1)
    fair = (1 / o[ok]) / (1 / o[ok]).sum(axis=1, keepdims=True)
    parts += [
        _rows("market", "ou25", fair[:, 0], tot[ok] >= 3),
        _rows("market", "ou25", fair[:, 1], tot[ok] < 3),
    ]
    if counts_preds is not None:
        c = counts_preds[counts_preds["season"].between(*seasons) & counts_preds["total"].notna()]
        for col in [x for x in c.columns if x.startswith(("p_team_", "p_base_"))]:
            kind, line = col.split("_")[1], float(col.rsplit("_", 1)[1])
            for stat, cs in c[c[col].notna()].groupby("stat"):
                # reds use the league-average model (the team model was worse, ADR 0008)
                if (stat == "reds") != (kind == "base"):
                    continue
                p, over = cs[col].to_numpy(), (cs["total"] > line).to_numpy()
                parts += [
                    _rows("model", f"{stat}_{line}", p, over),
                    _rows("model", f"{stat}_{line}", 1 - p, ~over),
                ]
    rows = pd.concat(parts, ignore_index=True)
    rows["band"] = (np.floor(rows["prob"] / BAND) * BAND).round(2)
    skill = {
        key: _skill(r["prob"].to_numpy(), r["hit"].to_numpy())
        for key, r in rows.groupby(["source", "market"])
    }
    table = (
        rows.groupby(["source", "market", "band"])
        .agg(n=("hit", "size"), predicted=("prob", "mean"), hit_rate=("hit", "mean"))
        .reset_index()
    )
    table["skill"] = [skill[(s, m)] for s, m in zip(table["source"], table["market"], strict=True)]
    gap = (table["hit_rate"] - table["predicted"]).abs()
    table["label"] = np.where(
        ~table["skill"] | (table["n"] < LOW_N) | (gap > LOW_GAP),
        "Low",
        np.where((table["n"] >= HIGH_N) & (gap <= HIGH_GAP), "High", "Medium"),
    )
    return table


def load_track_record(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def lookup(table: pd.DataFrame | None, source: str, market: str, prob: float) -> tuple[str, str]:
    """(label, record text) for one probability, e.g. ("High", "won 63% of 1,204")."""
    if table is None or np.isnan(prob):
        return "-", "no track record (run build-confidence)"
    band = round(float(np.floor(prob / BAND) * BAND), 2)
    hit = table[(table["source"] == source) & (table["market"] == market) & (table["band"] == band)]
    if hit.empty:
        return "Low", "no past predictions at this level"
    r = hit.iloc[0]
    return str(r["label"]), f"came true {r['hit_rate']:.0%} of {int(r['n']):,}"
