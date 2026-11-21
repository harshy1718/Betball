import numpy as np
import pandas as pd
import pytest

import football_edge.models.counts as counts_module
from football_edge.models.counts import CountParams, TeamCountModel


@pytest.fixture()
def corners_history():
    """Team A wins lots of corners; team D concedes lots."""
    rng = np.random.default_rng(0)
    own = {"A": 0.4, "B": 0.0, "C": 0.0, "D": -0.2}
    opp = {"A": -0.1, "B": 0.0, "C": 0.0, "D": 0.3}
    rows = []
    for w in range(80):
        d = pd.Timestamp("2023-08-01") + pd.Timedelta(days=7 * w)
        for h, a in (("A", "B"), ("C", "D")) if w % 2 else (("B", "C"), ("D", "A")):
            lh = np.exp(np.log(5) + 0.1 + own[h] + opp[a])
            la = np.exp(np.log(5) + own[a] + opp[h])
            rows.append(("E0", 2023, d, h, a, rng.poisson(lh), rng.poisson(la)))
    return pd.DataFrame(rows, columns=["league", "season", "date", "home", "away", "HC", "AC"])


def _fx(h, a):
    return pd.DataFrame(
        {"league": ["E0"], "date": [pd.Timestamp("2025-03-01")], "home": [h], "away": [a]}
    )


def test_team_effects_recovered(corners_history):
    m = TeamCountModel("corners")
    m.fit(corners_history)
    lam_ad, _ = m.expected(_fx("A", "D"))
    lam_dc, _ = m.expected(_fx("D", "C"))
    assert lam_ad[0] > 7 and lam_dc[0] < 5  # A against D wins far more corners
    base = TeamCountModel("corners", CountParams(team_effects=False))
    base.fit(corners_history)
    b_h, _ = base.expected(_fx("A", "D"))
    assert b_h[0] == pytest.approx(corners_history.HC.mean(), rel=0.1)


def test_p_over_is_a_valid_and_ordered_probability(corners_history):
    m = TeamCountModel("corners")
    m.fit(corners_history)
    ps = [m.p_over(_fx("A", "D"), line)[0] for line in (8.5, 9.5, 10.5, 11.5)]
    assert all(0 < p < 1 for p in ps) and ps == sorted(ps, reverse=True)


def test_unknown_team_gets_league_average(corners_history):
    m = TeamCountModel("corners")
    m.fit(corners_history)
    lam, _ = m.expected(_fx("New", "Other"))
    assert lam[0] == pytest.approx(np.exp(m.mu_l[0] + m.hfa[0]))


def test_gradient_matches_numeric(corners_history, monkeypatch):
    captured = {}
    real = counts_module.minimize

    def spy(fun, x0, **kw):
        captured["fun"], captured["x0"] = fun, x0
        return real(fun, x0, **kw)

    monkeypatch.setattr(counts_module, "minimize", spy)
    TeamCountModel("corners").fit(corners_history)
    fun = captured["fun"]
    th = captured["x0"] + np.random.default_rng(1).normal(0, 0.1, len(captured["x0"]))
    _, grad = fun(th)
    num = np.array(
        [(fun(th + 1e-6 * e)[0] - fun(th - 1e-6 * e)[0]) / 2e-6 for e in np.eye(len(th))]
    )
    np.testing.assert_allclose(grad, num, atol=1e-3)


def test_unknown_stat_rejected():
    with pytest.raises(ValueError):
        TeamCountModel("throw_ins")


def test_total_cards_are_yellows_plus_reds():
    from football_edge.models.counts import with_card_totals

    df = pd.DataFrame({"HY": [2, 1], "HR": [1, 0], "AY": [3, 0], "AR": [0, 0]})
    out = with_card_totals(df)
    assert out["HCARDS"].tolist() == [3, 1] and out["ACARDS"].tolist() == [3, 0]
