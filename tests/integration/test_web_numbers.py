"""The website must show exactly the numbers `football-edge predict` gives."""

import pandas as pd
import pytest

from football_edge.data.loader import load_matches
from football_edge.predict.fixtures import single_fixture
from football_edge.predict.pipeline import forecast_with_model
from football_edge.web.service import NoxService

pytestmark = pytest.mark.integration


def test_site_numbers_match_the_cli(cfg):
    svc = NoxService(cfg)
    if not svc.schedule_path.exists():
        pytest.skip("no schedule")
    sl = svc.slate()
    if sl.fc.empty:
        pytest.skip("no upcoming matches")
    f = sl.fc.iloc[0]
    history = load_matches(cfg)
    if history["date"].max() >= pd.Timestamp(f["date"]):
        pytest.skip("results newer than the fixture; training sets would differ")
    known = {str(lg): set(g["home"]) | set(g["away"]) for lg, g in history.groupby("league")}
    fx = single_fixture(str(f["home"]), str(f["away"]), known, cfg.leagues,
                        league=str(f["league"]), date=f"{pd.Timestamp(f['date']):%Y-%m-%d}",
                        aliases_path=svc.aliases)  # fmt: skip
    cli, _, _ = forecast_with_model(cfg, history, fx)
    site = svc.match(str(f["id"]))
    assert site["probs"]["home"] == pytest.approx(float(cli.iloc[0]["p_h"]), abs=1e-9)
    assert site["probs"]["away"] == pytest.approx(float(cli.iloc[0]["p_a"]), abs=1e-9)
    assert site["top_score"]["score"] == cli.iloc[0]["top_score"]
    assert sum(map(sum, site["grid"])) > 0.95  # the 0-5 grid holds nearly all the mass
