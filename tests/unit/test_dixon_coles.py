import numpy as np
import pandas as pd
import pytest

import football_edge.models.dixon_coles as dc_module
from football_edge.models.dixon_coles import DixonColes, DixonColesParams, tau


@pytest.fixture()
def history():
    # One league, two seasons. A thrashes everyone, D loses to everyone.
    # Season 2016 adds promoted team P (replacing D), which loses heavily.
    rng = np.random.default_rng(0)
    strength = {"A": 1.0, "B": 0.0, "C": 0.0, "D": -0.8, "P": -0.8}
    rows = []
    day = pd.Timestamp("2015-08-01")
    for season, teams in [(2015, "ABCD"), (2016, "ABCP")]:
        for rnd in range(30):
            d = day + pd.Timedelta(days=7 * rnd + 365 * (season - 2015))
            for h in teams:
                for a in teams:
                    if h == a or rng.random() > 0.3:
                        continue
                    lh = np.exp(0.2 + 0.25 + strength[h] - strength[a] / 2)
                    la = np.exp(0.2 + strength[a] - strength[h] / 2)
                    rows.append(("E0", season, d, h, a, rng.poisson(lh), rng.poisson(la)))
    df = pd.DataFrame(rows, columns=["league", "season", "date", "home", "away", "FTHG", "FTAG"])
    df["FTR"] = np.select([df.FTHG > df.FTAG, df.FTHG < df.FTAG], ["H", "A"], "D")
    return df


def _fixture(home, away, season=2016, date="2017-06-01"):
    return pd.DataFrame(
        {
            "league": ["E0"],
            "season": [season],
            "date": [pd.Timestamp(date)],
            "home": [home],
            "away": [away],
        }
    )


def test_tau_only_changes_low_scores():
    x = np.array([0, 0, 1, 1, 2])
    y = np.array([0, 1, 0, 1, 2])
    t = tau(x, y, np.full(5, 1.5), np.full(5, 1.0), rho=-0.1)
    np.testing.assert_allclose(t, [1.15, 0.85, 0.9, 1.1, 1.0])


def test_score_matrix_is_a_distribution(history):
    m = DixonColes()
    m.fit(history)
    grid = m.score_matrix(1.6, 1.1)
    assert grid.shape == (11, 11)
    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_probabilities_valid_and_strong_team_favoured(history):
    m = DixonColes()
    m.fit(history)
    test = pd.concat([_fixture("A", "C"), _fixture("C", "A")], ignore_index=True)
    p = m.predict_proba(test)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] > 0.5  # A at home to C
    assert p[1, 2] > p[1, 0]  # A away at C still more likely to win than lose


def test_prediction_never_reads_results(history):
    m = DixonColes()
    m.fit(history)
    test = _fixture("A", "B")
    p1 = m.predict_proba(test)
    with_results = test.assign(FTHG=[0], FTAG=[9], FTR=["A"])
    p2 = m.predict_proba(with_results)
    np.testing.assert_allclose(p1, p2)


def test_unseen_team_gets_promoted_prior(history):
    m = DixonColes()
    m.fit(history)
    assert m._prior_att < 0  # promoted sides score less than average
    lam_new, _ = m.expected_goals(_fixture("NEW", "B", season=2017, date="2017-08-10"))
    lam_c, _ = m.expected_goals(_fixture("C", "B", season=2017, date="2017-08-10"))
    assert lam_new[0] < lam_c[0]


def test_relegated_team_returning_is_treated_as_newcomer(history):
    # D was last seen in 2015; in 2017 it is not "active", so it gets the prior.
    m = DixonColes()
    m.fit(history)
    assert m._rating("E0", "D", 2017) == (m._prior_att, m._prior_def)


def test_analytic_gradient_matches_numeric(history, monkeypatch):
    captured = {}
    real_minimize = dc_module.minimize

    def spy(fun, x0, **kwargs):
        captured["fun"], captured["x0"] = fun, x0
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(dc_module, "minimize", spy)
    DixonColes(DixonColesParams(ridge=1.0)).fit(history)
    fun = captured["fun"]
    rng = np.random.default_rng(1)
    theta = captured["x0"] + rng.normal(0, 0.1, len(captured["x0"]))
    theta[3] = 0.05  # rho (one league: mu, hfa, c, rho)
    _, grad = fun(theta)
    eps = 1e-6
    num = np.array(
        [
            (fun(theta + eps * e)[0] - fun(theta - eps * e)[0]) / (2 * eps)
            for e in np.eye(len(theta))
        ]
    )
    np.testing.assert_allclose(grad, num, atol=1e-4)


def _with_shots(history, unlucky="B"):
    # Every side gets 4 shots on target, except `unlucky`, which gets 9 but scores the same.
    df = history.copy()
    df["HST"] = np.where(df.home == unlucky, 9, 4)
    df["AST"] = np.where(df.away == unlucky, 9, 4)
    return df


def test_shots_weight_zero_ignores_shots(history):
    plain = DixonColes(DixonColesParams(shots_weight=0.0))
    plain.fit(history)
    shots = DixonColes(DixonColesParams(shots_weight=0.0))
    shots.fit(_with_shots(history))
    np.testing.assert_allclose(plain.attack, shots.attack)


def test_shots_raise_rating_of_team_creating_chances(history):
    goals_only = DixonColes(DixonColesParams(shots_weight=0.0))
    goals_only.fit(_with_shots(history))
    with_shots = DixonColes(DixonColesParams(shots_weight=1.0))
    with_shots.fit(_with_shots(history))
    b = with_shots._team_idx[("E0", "B")]
    assert with_shots.attack[b] > goals_only.attack[b]


def test_gradient_correct_with_shots(history, monkeypatch):
    captured = {}
    real_minimize = dc_module.minimize

    def spy(fun, x0, **kwargs):
        captured["fun"], captured["x0"] = fun, x0
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(dc_module, "minimize", spy)
    DixonColes(DixonColesParams(shots_weight=0.5)).fit(_with_shots(history))
    fun = captured["fun"]
    theta = captured["x0"] + np.random.default_rng(2).normal(0, 0.1, len(captured["x0"]))
    theta[3] = 0.05
    _, grad = fun(theta)
    num = np.array(
        [(fun(theta + 1e-6 * e)[0] - fun(theta - 1e-6 * e)[0]) / 2e-6 for e in np.eye(len(theta))]
    )
    np.testing.assert_allclose(grad, num, atol=1e-4)
