"""Measure the extractor on hand-labelled articles before trusting it.

Labels file: JSON lines, one article each:
  {"id": "...", "article": "...",
   "expected": {"players": [{"team": "...", "player": "...", "status": "out"}],
                "manager_change": [], "rotation_expected": []}}

Scores:
  * player facts: precision (share of extracted facts that are right) and recall (share of
    true facts found). A fact is right if team, player surname and status all match.
  * valid answers: share of articles where the model produced a usable answer.
  * latency per article and tokens used (cost = tokens x your provider's price).
Target before use (docs/news.md): precision >= 0.90 (wrong "out" calls are the costly
error), recall >= 0.70, on at least 100 labelled articles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from football_edge.news.extract import ChatClient, extract
from football_edge.news.impact import _norm


@dataclass(frozen=True)
class EvalResult:
    articles: int
    valid_rate: float
    precision: float
    recall: float
    true_pos: int
    extracted: int
    expected: int
    mean_latency_s: float
    tokens: int
    errors: tuple[str, ...]


def _key(team: str, player: str, status: str) -> tuple[str, str, str]:
    surname = _norm(player).split()[-1] if player.strip() else ""
    return _norm(team), surname, status.strip().lower()


def load_labels(path: Path) -> list[dict[str, object]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            if not {"article", "expected"} <= set(row):
                raise ValueError(f"{path.name} line {n}: needs 'article' and 'expected'")
            rows.append(row)
    return rows


def evaluate(client: ChatClient, labels: list[dict[str, object]]) -> EvalResult:
    tp = n_ext = n_exp = valid = tokens = 0
    latency = 0.0
    errors: list[str] = []
    for row in labels:
        article = str(row["article"])
        expected = row["expected"]
        assert isinstance(expected, dict)
        truth = {_key(p["team"], p["player"], p["status"]) for p in expected.get("players", [])}
        res = extract(client, article)
        latency += res.latency_s
        tokens += res.tokens
        n_exp += len(truth)
        if res.facts is None:
            errors.append(f"{row.get('id', '?')}: {res.error}")
            continue
        valid += 1
        got = {_key(p.team, p.player, p.status) for p in res.facts.players}
        n_ext += len(got)
        tp += len(got & truth)
    n = max(len(labels), 1)
    return EvalResult(
        articles=len(labels),
        valid_rate=valid / n,
        precision=tp / n_ext if n_ext else float("nan"),
        recall=tp / n_exp if n_exp else float("nan"),
        true_pos=tp,
        extracted=n_ext,
        expected=n_exp,
        mean_latency_s=latency / n,
        tokens=tokens,
        errors=tuple(errors),
    )
