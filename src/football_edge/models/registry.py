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
    if name == "glicko":
        from football_edge.models.glicko import Glicko, GlickoParams

        return lambda: Glicko(GlickoParams(**params))
    if name == "bivariate_poisson":  # ADR 0025: DC ratings, bivariate Poisson scores
        from football_edge.models.bivariate import BivariatePoissonDC

        bp_params = DixonColesParams.from_dict(params)
        return lambda: BivariatePoissonDC(bp_params)
    if name == "dixon_coles_per_league":  # ADR 0023: base settings + per-league overrides
        from football_edge.models.per_league import PerLeagueDixonColes

        base = {k: v for k, v in params.items() if k != "by_league"}
        by = dict(params.get("by_league") or {})
        return lambda: PerLeagueDixonColes(base, by)
    if name.startswith("dixon_coles"):
        dc_params = DixonColesParams.from_dict(params)
        return lambda: _named(DixonColes(dc_params), name)
    raise ValueError(f"unknown model: {name!r} (available: team_blind, elo, dixon_coles*)")


def _named(model: DixonColes, name: str) -> DixonColes:
    model.name = name
    return model
