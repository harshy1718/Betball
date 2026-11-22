from typing import ClassVar

import pandas as pd
import pytest

from football_edge.web.service import MatchNotFoundError, match_id, mood_of, nox_line


def test_mood_sure_only_when_more_likely_than_not_and_not_low():
    assert mood_of(0.53, "High") == "sure"
    assert mood_of(0.53, "Medium") == "sure"
    assert mood_of(0.61, "Low") == "unsure"  # the track record says don't trust it
    assert mood_of(0.48, "High") == "unsure"  # nothing is more likely than not


def test_nox_line_quotes_only_the_pick_probability():
    assert nox_line("Liverpool", 0.531, "sure", "High") == "Liverpool, 53%. Confident on this one."
    assert nox_line("Liverpool", 0.531, "sure", "Medium") == "Liverpool, 53%. That's my lean."
    assert nox_line("Draw", 0.34, "unsure", "Low") == "A draw 34%, but this could go any way."
    low = nox_line("Osasuna", 0.61, "unsure", "Low")
    assert low == "Osasuna 61%, but I'm not sure of this one."


def test_match_id_is_stable_and_url_safe():
    k = pd.Timestamp("2026-11-22 16:30", tz="UTC")
    mid = match_id("E0", k, "Nott'm Forest", "Man United")
    assert mid == "e0-20261122-nott-m-forest-man-united"


class StubService:
    """Stands in for NoxService so the routes can be tested without data."""

    row: ClassVar[dict] = {"id": "e0-1", "home": "Liverpool", "away": "Man United", "golden": []}

    def meta(self):
        return {"trained_up_to": "08 Nov 2026", "matches": 1, "days": 7, "warning": None}

    def next_matches(self, n=5):
        return [self.row][:n]

    def matches(self, league=None):
        return [self.row] if league in (None, "E0") else []

    def match(self, mid):
        if mid != "e0-1":
            raise MatchNotFoundError(mid)
        return {**self.row, "grid": [[0.1]]}

    def golden_calls(self):
        raise FileNotFoundError("no schedule yet: run `football-edge schedule`")


@pytest.fixture()
def client(cfg, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from football_edge.web.app import create_app

    return TestClient(create_app(cfg, StubService(), dist=tmp_path))  # type: ignore[arg-type]


def test_routes(client):
    assert client.get("/api/meta").json()["trained_up_to"] == "08 Nov 2026"
    assert client.get("/api/next?n=5").json()[0]["home"] == "Liverpool"
    assert client.get("/api/matches?league=I1").json() == []
    assert client.get("/api/matches/e0-1").json()["grid"] == [[0.1]]


def test_errors_are_http_errors_not_crashes(client):
    assert client.get("/api/matches/nope").status_code == 404
    r = client.get("/api/golden")
    assert r.status_code == 503 and "schedule" in r.json()["detail"]


def test_serves_the_built_app_with_client_side_routes(cfg, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from football_edge.web.app import create_app

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    c = TestClient(create_app(cfg, StubService(), dist=tmp_path))  # type: ignore[arg-type]
    assert "root" in c.get("/").text
    assert "root" in c.get("/match/e0-1").text  # a React route, not a file
    assert c.get("/api/nothing").status_code == 404
    assert c.get("/../pyproject.toml").status_code in (200, 404)
    assert "build-system" not in c.get("/..%2Fpyproject.toml").text
