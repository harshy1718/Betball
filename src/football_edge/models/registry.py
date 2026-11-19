"""Build a model by name with parameters from the config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from football_edge.models.base import Model
from football_edge.models.dixon_coles import DixonColes, DixonColesParams
from football_edge.models.elo import Elo, EloParams
from football_edge.models.team_blind import TeamBlind


def model_factory(name: str, params: dict[str, Any] | None = None) -> Callable[[], Model]:
    """Return a function that builds a fresh, unfitted model (one per fold)."""
    params = params or {}
    if name == "team_blind":
        return TeamBlind
    if name == "elo":
        elo_params = EloParams.from_dict(params)
        return lambda: Elo(elo_params)
    if name == "dixon_coles":
        dc_params = DixonColesParams.from_dict(params)
        return lambda: DixonColes(dc_params)
    raise ValueError(f"unknown model: {name!r} (available: team_blind, elo, dixon_coles)")
