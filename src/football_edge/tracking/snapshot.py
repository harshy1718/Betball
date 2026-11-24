"""Write what the model and the desk say now into the logs, before kickoff.

Two logs (see ledger.py):
  * predictions (committed to git): every leg of every match not yet kicked off, with the
    model's chance and confidence. Only matches whose numbers changed since their last
    logged block are written, so each match gets a new block only when the model moves.
  * desk slips (private, under bets/): the board Nox generated (top of the board, and per
    match the singles and the safe and risky parlays), again only when a match's slips change.
Grading always uses the last block logged before kickoff.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from football_edge.config import Config
from football_edge.tracking.ledger import Ledger

PRED_COLS = [
    "taken_at", "match_id", "league", "kickoff_utc", "home_key", "away_key",
    "market", "sel", "line", "p", "confidence", "skill",
]  # fmt: skip
SLIP_COLS = ["taken_at", "group", "kickoff_utc", "title", "kind", "keys", "labels", "p", "risk"]
POSTED_COLS = [
    "taken_at", "match_id", "league", "kickoff_utc", "home_key", "away_key", "home", "away",
    "kind", "market", "sel", "line", "call", "p",
]  # fmt: skip


def ledgers(cfg: Config) -> dict[str, Ledger]:
    t = cfg.tracking
    return {
        "predictions": Ledger(
            cfg.root / t.get("predictions_dir", "records/predictions"), PRED_COLS
        ),
        "slips": Ledger(cfg.root / t.get("slips_dir", "bets/desk_log"), SLIP_COLS),
        "posted": Ledger(cfg.root / t.get("posted_dir", "records/posted"), POSTED_COLS),
    }


def _sig(df: pd.DataFrame, cols: list[str]) -> str:
    text = df[cols].astype(str).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _latest_sigs(old: pd.DataFrame, group: str, cols: list[str]) -> dict[str, str]:
    if old.empty:
        return {}
    last = old[old["taken_at"] == old.groupby(group)["taken_at"].transform("max")]
    return {str(k): _sig(g, cols) for k, g in last.groupby(group)}


def prediction_rows(matches: list[dict[str, Any]], legs: dict[str, list[Any]],
                    now: pd.Timestamp) -> pd.DataFrame:  # fmt: skip
    rows = []
    for m in matches:
        if pd.Timestamp(m["kickoff_utc"]) <= now:
            continue
        for x in legs.get(m["id"], []):
            rows.append({
                "match_id": x.match_id, "league": x.league, "kickoff_utc": x.kickoff_utc,
                "home_key": x.home_key, "away_key": x.away_key, "market": x.market,
                "sel": x.sel, "line": "" if x.line is None else f"{x.line:g}",
                "p": f"{x.p:.4f}", "confidence": x.confidence, "skill": str(bool(x.skill)),
            })  # fmt: skip
    return pd.DataFrame(rows, columns=PRED_COLS[1:])


def slip_rows(board: dict[str, Any]) -> pd.DataFrame:
    rows = []

    def add(group: str, s: dict[str, Any]) -> None:
        rows.append({
            "group": group,
            "kickoff_utc": min(x["kickoff_utc"] for x in s["legs"]),
            "title": s["title"], "kind": s["kind"],
            "keys": ";".join(x["key"] for x in s["legs"]),
            "labels": " + ".join(f"{x['label']} ({x['home']} v {x['away']})" for x in s["legs"]),
            "p": f"{s['p']:.4f}", "risk": s["risk"],
        })  # fmt: skip

    for s in board.get("top", []):
        add("top", s)
    for m in board.get("matches", []):
        for s in m["singles"]:
            add(m["match"]["id"], s)
        for s in m["parlays"].values():
            if s:
                add(m["match"]["id"], s)
    return pd.DataFrame(rows, columns=SLIP_COLS[1:])


def take_snapshot(cfg: Config, desk: Any, now: pd.Timestamp | None = None) -> dict[str, int]:
    """Log every changed prediction and desk slip. Returns rows written per log."""
    now = now or pd.Timestamp.now(tz="UTC")
    taken_at = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    books = ledgers(cfg)
    out = {"predictions": 0, "slips": 0}

    matches, legs = desk.legs_by_match()
    preds = prediction_rows(matches, legs, now)
    if len(preds):
        cols = ["market", "sel", "line", "p", "confidence"]
        old = _latest_sigs(books["predictions"].read(), "match_id", cols)
        changed = [
            mid for mid, g in preds.groupby("match_id") if old.get(str(mid)) != _sig(g, cols)
        ]
        out["predictions"] = books["predictions"].append(
            preds[preds["match_id"].isin(changed)], taken_at
        )

    slips = slip_rows(desk.board(now))
    if len(slips):
        cols = ["title", "keys", "p"]
        old = _latest_sigs(books["slips"].read(), "group", cols)
        changed = [g for g, d in slips.groupby("group") if old.get(str(g)) != _sig(d, cols)]
        out["slips"] = books["slips"].append(slips[slips["group"].isin(changed)], taken_at)
    return out
