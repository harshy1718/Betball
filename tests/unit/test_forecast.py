import numpy as np
import pandas as pd
import pytest

from football_edge.predict.forecast import fit_for, forecast


def _history():
    rng = np.random.default_rng(1)
    teams = "ABCDEF"
    rows = []
    for w in range(60):
        d = pd.Timestamp("2024-08-10") + pd.Timedelta(days=7 * w)
        for h, a in zip(teams[::2], teams[1::2], strict=True):
            if w % 2:
                h, a = a, h
            rows.append(("E0", 2024, d, h, a, rng.poisson(1.5), rng.poisson(1.1)))
    df = pd.DataFrame(rows, columns=["league", "season", "date", "home", "away", "FTHG", "FTAG"])
    df["FTR"] = np.select([df.FTHG > df.FTAG, df.FTHG < df.FTAG], ["H", "A"], "D")
    return df


def test_forecast_is_consistent_and_uses_only_the_past():
    hist = _history()
    fx = pd.DataFrame(
        {
            "league": ["E0"],
            "season": [2024],
            "date": [pd.Timestamp("2025-03-01")],
            "home": ["A"],
            "away": ["B"],
        }
    )
    model, info = fit_for(hist, fx, {"xi": 0.0025, "ridge": 1.0, "max_goals": 10})
    assert info.last_match < fx["date"].min()
    assert info.n_matches == int((hist["date"] < "2025-03-01").sum())
    out = forecast(model, fx).iloc[0]
    assert np.isclose(out.p_h + out.p_d + out.p_a, 1.0)
    assert np.isclose(out.p_over25 + out.p_under25, 1.0)
    assert np.allclose([out.p_h, out.p_d, out.p_a], model.predict_proba(fx)[0])
    assert out.top_scores.count("%") == 3


def test_market_view_prefers_sharp_prices_and_removes_margin():
    from football_edge.predict.forecast import market_view

    fc = pd.DataFrame(
        {
            "p_h": [0.5, 0.5],
            "p_d": [0.3, 0.3],
            "p_a": [0.2, 0.2],
            "p_over25": [0.5, 0.5],
            "p_under25": [0.5, 0.5],
            "p_btts_yes": [0.5, 0.5],
            "p_btts_no": [0.5, 0.5],
            "odds_h": [1.8, 1.8],
            "odds_d": [3.6, 3.6],
            "odds_a": [4.5, 4.5],
            "odds_over25": [np.nan, 1.9],
            "odds_under25": [np.nan, 1.9],
            "odds_btts_yes": [np.nan, np.nan],
            "odds_btts_no": [np.nan, np.nan],
            "ref_h": [2.0, np.nan],
            "ref_d": [4.0, np.nan],
            "ref_a": [4.0, np.nan],
            "ref_over25": [np.nan, np.nan],
            "ref_under25": [np.nan, np.nan],
        }
    )
    out = market_view(fc)
    assert out["mkt_src_1x2"].tolist() == ["sharp", "site"]
    assert out.loc[0, ["mkt_h", "mkt_d", "mkt_a"]].tolist() == pytest.approx([0.5, 0.25, 0.25])
    site = np.array([1 / 1.8, 1 / 3.6, 1 / 4.5])
    assert out.loc[1, ["mkt_h", "mkt_d", "mkt_a"]].tolist() == pytest.approx(site / site.sum())
    assert out["mkt_src_ou25"].tolist() == ["-", "site"] and np.isnan(out.loc[0, "mkt_over25"])
    assert out["mkt_src_btts"].tolist() == ["-", "-"]
