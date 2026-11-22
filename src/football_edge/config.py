"""Typed, immutable configuration loaded from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OddsSource:
    name: str
    cols: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    first_test_season: int
    last_test_season: int
    min_train_matches: int
    refit_frequency: str


@dataclass(frozen=True)
class Config:
    root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    leagues: tuple[str, ...]
    closing_priority: tuple[OddsSource, ...]
    pre_priority: tuple[OddsSource, ...]
    max_pre_priority: tuple[OddsSource, ...]
    max_close_priority: tuple[OddsSource, ...]
    ou_close_priority: tuple[OddsSource, ...]
    evaluation: EvaluationConfig
    models: dict[str, dict[str, Any]]
    tuning: dict[str, dict[str, Any]]
    predict: dict[str, Any]
    staking: dict[str, Any]
    news: dict[str, Any]
    writer: dict[str, Any]
    content: dict[str, Any]
    analyst: dict[str, Any]

    @property
    def matches_path(self) -> Path:
        return self.processed_dir / "matches.parquet"


def _sources(items: list[dict[str, Any]]) -> tuple[OddsSource, ...]:
    return tuple(OddsSource(i["name"], tuple(i["cols"])) for i in items)


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("FOOTBALL_EDGE_CONFIG", "configs/default.yaml"))
    root = (root or cfg_path.resolve().parent.parent).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ev = raw["evaluation"]
    return Config(
        root=root,
        raw_dir=root / raw["paths"]["raw"],
        processed_dir=root / raw["paths"]["processed"],
        reports_dir=root / raw["paths"]["reports"],
        leagues=tuple(raw["leagues"]),
        closing_priority=_sources(raw["odds"]["closing_priority"]),
        pre_priority=_sources(raw["odds"]["pre_priority"]),
        max_pre_priority=_sources(raw["odds"].get("max_pre_priority", [])),
        max_close_priority=_sources(raw["odds"].get("max_close_priority", [])),
        ou_close_priority=_sources(raw["odds"].get("ou_close_priority", [])),
        evaluation=EvaluationConfig(
            ev["first_test_season"],
            ev["last_test_season"],
            ev["min_train_matches"],
            ev["refit_frequency"],
        ),
        models={name: dict(params or {}) for name, params in raw.get("models", {}).items()},
        tuning={name: dict(t or {}) for name, t in raw.get("tuning", {}).items()},
        predict=dict(raw.get("predict") or {}),
        staking=dict(raw.get("staking") or {}),
        news=dict(raw.get("news") or {}),
        writer=dict(raw.get("writer") or {}),
        content=dict(raw.get("content") or {}),
        analyst=dict(raw.get("analyst") or {}),
    )
