"""Calibration maps (ADR 0020), the score grid's tail and validity checks, the log's new
columns, and the referee option of the count model."""

import numpy as np
import pandas as pd
import pytest

from football_edge.models import calibration as cal
from football_edge.models.dixon_coles import InvalidGridError, score_grid, score_grids
from football_edge.tracking.ledger import Ledger


def test_identity_map_changes_nothing():
    lam, mu = np.array([1.4, 0.6, 2.9]), np.array([1.1, 1.8, 0.4])
    lam2, mu2 = cal.GoalCalibration(t_ref=0.3).expected(lam, mu)
    assert np.allclose(lam2, lam) and np.allclose(mu2, mu)
    g = score_grid(1.4, 1.1, -0.03)
    assert np.allclose(cal.adjust_draws(g, 0.0), g)


def test_map_stretches_the_gap_and_pulls_totals_together():
    c = cal.GoalCalibration(a=1.2, d=0.5, t_ref=0.0)
    lam, mu = np.array([2.0, 0.8]), np.array([0.5, 0.9])
    lam2, mu2 = c.expected(lam, mu)
    assert lam2[0] / mu2[0] == pytest.approx((2.0 / 0.5) ** 1.2)
    assert lam2[0] * mu2[0] == pytest.approx((2.0 * 0.5) ** 0.5)


def test_draw_adjustment_keeps_a_distribution():
    g = cal.adjust_draws(score_grid(1.3, 1.0, -0.03), 0.15)
    assert g.sum() == pytest.approx(1.0)
    assert np.trace(g) > np.trace(score_grid(1.3, 1.0, -0.03))
    many = cal.adjust_draws(score_grids(np.array([1.3, 2.0]), np.array([1.0, 0.7]), -0.03), 0.15)
    assert np.allclose(many[0], g)


def test_fit_recovers_a_known_miscalibration():
    """Simulate scores from expected goals whose total is twice as spread as the truth: the
    fitted map must pull totals together (d < 1)."""
    rng = np.random.default_rng(1)
    t_true = rng.normal(0.55, 0.15, 6000)
    s = rng.normal(0.2, 0.5, 6000)
    lam_t, mu_t = np.exp((t_true + s) / 2), np.exp((t_true - s) / 2)
    h, a = rng.poisson(lam_t), rng.poisson(mu_t)
    t_model = 0.55 + 2 * (t_true - 0.55)  # the model exaggerates totals
    lam, mu = np.exp((t_model + s) / 2), np.exp((t_model - s) / 2)
    fit = cal.fit_goals(lam, mu, h, a, rho=0.0)
    assert 0.35 < fit.d < 0.7
    raw = cal.goal_scores(cal._grids_for(cal.GoalCalibration(), lam, mu, 0.0), h, a)
    new = cal.goal_scores(cal._grids_for(fit, lam, mu, 0.0), h, a)
    assert new["ou25"] < raw["ou25"]


def test_walk_forward_never_uses_the_season_it_calibrates():
    rng = np.random.default_rng(2)
    rows = []
    for season in (2018, 2019, 2020):
        lam, mu = rng.uniform(0.8, 2.2, 400), rng.uniform(0.6, 1.8, 400)
        rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "exp_home": lam,
                    "exp_away": mu,
                    "goals_home": rng.poisson(lam),
                    "goals_away": rng.poisson(mu),
                    "p_h": 0.4,
                    "p_d": 0.3,
                    "p_a": 0.3,
                    "p_over25": 0.5,
                    "p_btts": 0.5,
                }
            )
        )
    preds = pd.concat(rows, ignore_index=True)
    out, report = cal.calibrate_goals_walk_forward(preds, rho=0.0, first=2019)
    assert set(out["season"]) == {2019, 2020} and list(report["season"]) == [2019, 2020]
    assert report.loc[0, "fitted_on"].startswith("2018-2018")
    assert report.loc[1, "fitted_on"].startswith("2018-2019")
    assert {"p_h_raw", "exp_home_raw", "cal_delta"} <= set(out.columns)


def test_cards_map_pulls_towards_the_league_average():
    c = cal.CardCalibration(d=0.5, f=2.0)
    assert c.mean(np.array([6.0]), np.array([4.0]))[0] == pytest.approx(np.sqrt(24))
    assert c.k(np.array([10.0]))[0] == 20.0
    p = cal.p_over_nb(np.array([4.0, 4.0]), np.array([np.inf, 5.0]), 4.5)
    assert 0 < p[0] < 1 and p[1] > p[0] - 0.2


def test_save_and_load_only_adopted_maps(tmp_path):
    path = tmp_path / "c.json"
    cal.save(
        path,
        cal.GoalCalibration(a=1.1),
        cal.CardCalibration(d=0.6),
        {"goals": True, "cards": False},
    )
    g, c = cal.load(path)
    assert g is not None and g.a == 1.1 and c is None
    assert cal.load(tmp_path / "missing.json") == (None, None)


def test_grid_folds_the_tail_into_the_edge_instead_of_spreading_it():
    g = score_grid(4.2, 0.9, -0.03)
    assert g.sum() == pytest.approx(1.0)

    # P(home scores 10 or more) is in the edge row, not spread over the other scores
    from scipy.stats import poisson

    assert g[10].sum() == pytest.approx(1 - poisson.cdf(9, 4.2), rel=1e-3)


def test_invalid_low_score_correction_is_an_error_not_a_clip():
    with pytest.raises(InvalidGridError):
        score_grid(3.0, 3.0, 0.2)  # 1 - lam*mu*rho < 0
    with pytest.raises(InvalidGridError):
        score_grids(np.array([1.0, -1.0]), np.array([1.0, 1.0]), 0.0)


def test_log_gains_columns_without_rewriting_old_files(tmp_path):
    old = Ledger(tmp_path, ["taken_at", "x"])
    old.append(pd.DataFrame({"x": ["1"]}), "2026-11-24T10:00:00Z")
    new = Ledger(tmp_path, ["taken_at", "x", "y"])
    new.append(pd.DataFrame({"x": ["2"], "y": ["b"]}), "2026-11-25T10:00:00Z")
    assert sorted(p.name for p in new.files()) == ["2026-11.csv", "2026-11_v2.csv"]
    df = new.read()
    assert list(df["y"]) == ["", "b"] and not new.verify()


def test_referee_effect_learns_a_strict_referee():
    rng = np.random.default_rng(3)
    from football_edge.models.counts import CountParams, TeamCountModel

    n = 1500
    teams = [f"T{i}" for i in range(10)]
    ref = rng.choice(["Strict", "Easy"], n)
    boost = np.where(ref == "Strict", 1.6, 1.0)
    df = pd.DataFrame(
        {
            "league": "E0",
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "home": rng.choice(teams, n),
            "away": rng.choice(teams, n),
            "HY": rng.poisson(1.8 * boost),
            "AY": rng.poisson(2.0 * boost),
            "HR": 0,
            "AR": 0,
            "referee": ref,
        }
    )
    m = TeamCountModel("cards", CountParams(referee_effects=True))
    m.fit(df)
    test = df.head(2).assign(referee=["Strict", "Easy"])
    lam, mu = m.expected(test.assign(home="T0", away="T1"))
    assert (lam + mu)[0] > 1.3 * (lam + mu)[1]
