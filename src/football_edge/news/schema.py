"""The fixed form the LLM must fill, and strict validation of what it returns."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

STATUSES = ("out", "doubtful", "suspended", "returning")
# Probability the player misses the match, per status. "returning" = back from absence.
MISS_PROB = {"out": 1.0, "suspended": 1.0, "doubtful": 0.5, "returning": 0.0}


class SchemaError(ValueError):
    """Raised when the model's answer does not fit the form."""


@dataclass(frozen=True)
class PlayerNews:
    team: str
    player: str
    status: str  # one of STATUSES
    reason: str  # "hamstring", "red card", "rest"... free text, for humans
    evidence: str  # the sentence in the article that supports this fact


@dataclass(frozen=True)
class ArticleFacts:
    players: tuple[PlayerNews, ...]
    manager_change: tuple[str, ...]  # teams whose manager has just left or been replaced
    rotation_expected: tuple[str, ...]  # teams expected to rest players (e.g. cup next)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


JSON_SCHEMA_TEXT = """{
  "players": [
    {"team": "<club>", "player": "<full name>",
     "status": "out|doubtful|suspended|returning",
     "reason": "<short reason>", "evidence": "<exact sentence from the article>"}
  ],
  "manager_change": ["<club>"],
  "rotation_expected": ["<club>"]
}"""


def _strip_fences(text: str) -> str:
    """Models often wrap JSON in ```json fences; take the outermost {...} block."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise SchemaError("no JSON object in the answer")
    return text[start : end + 1]


def _str_list(raw: Any, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise SchemaError(f"{field} must be a list of strings")
    return tuple(x.strip() for x in raw if x.strip())


def parse_answer(text: str, article: str) -> ArticleFacts:
    """Validate the model's answer. Facts whose evidence is not in the article are dropped:
    a fact the model cannot quote is treated as made up."""
    try:
        raw = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        raise SchemaError(f"invalid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise SchemaError("answer must be a JSON object")
    players = []
    norm_article = " ".join(article.split()).lower()
    for p in raw.get("players") or []:
        if not isinstance(p, dict):
            raise SchemaError("each player entry must be an object")
        status = str(p.get("status", "")).strip().lower()
        if status not in STATUSES:
            raise SchemaError(f"bad status {status!r} (allowed: {', '.join(STATUSES)})")
        team, player = str(p.get("team", "")).strip(), str(p.get("player", "")).strip()
        if not team or not player:
            raise SchemaError("player entries need team and player")
        evidence = str(p.get("evidence", "")).strip()
        if not evidence or " ".join(evidence.split()).lower() not in norm_article:
            continue  # not supported by the text
        players.append(PlayerNews(team, player, status, str(p.get("reason", "")).strip(), evidence))
    return ArticleFacts(
        players=tuple(players),
        manager_change=_str_list(raw.get("manager_change"), "manager_change"),
        rotation_expected=_str_list(raw.get("rotation_expected"), "rotation_expected"),
    )
