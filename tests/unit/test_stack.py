import numpy as np
import pandas as pd
import pytest

from football_edge.markets.stack import blend, fit_beta, stack_test


def _written() -> bool:
    try:
        blend(np.full((1, 3), 1 / 3), np.full((1, 3), 1 / 3), 0.5)
    except NotImplementedError:
        return False
    return True


needs_blend = pytest.mark.skipif(
    not _written(), reason="blend not written yet (docs/A4_2_blend_spec.md)"
)

MARKET = np.array([[0.476, 0.2455, 0.2785]])  # Spain v England, margin removed
MODEL = np.array([[0.55, 0.22, 0.23]])


@needs_blend
def test_beta_zero_is_market_and_one_is_model():
    np.testing.assert_allclose(blend(MARKET, MODEL, 0.0), MARKET, atol=1e-12)
    np.testing.assert_allclose(blend(MARKET, MODEL, 1.0), MODEL, atol=1e-12)


@needs_blend
def test_worked_example_beta_half():
    raw = np.sqrt(MARKET * MODEL)  # market * (model/market) ** 0.5
    expected = raw / raw.sum()
    np.testing.assert_allclose(blend(MARKET, MODEL, 0.5), expected, atol=1e-12)
    assert 0.476 < blend(MARKET, MODEL, 0.5)[0, 0] < 0.55


@needs_blend
def test_rows_sum_to_one_for_any_beta():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([2, 1, 1], size=50)
    b = rng.dirichlet([1, 1, 2], size=50)
    for beta in (-0.5, 0.3, 2.0):
        np.testing.assert_allclose(blend(a, b, beta).sum(axis=1), 1.0)


@needs_blend
def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        blend(MARKET, np.vstack([MODEL, MODEL]), 0.5)


@needs_blend
def test_inputs_not_modified():
    m, p = MARKET.copy(), MODEL.copy()
    blend(m, p, 0.7)
    np.testing.assert_array_equal(m, MARKET)
    np.testing.assert_array_equal(p, MODEL)


def _simulated(n_weeks=120, informative=True, seed=0):
    """Truth = market tilted by a signal; the 'model' sees the signal only if informative."""
    rng = np.random.default_rng(seed)
    n = n_weeks * 10
    market = rng.dirichlet([4, 2.5, 3], size=n)
    signal = rng.normal(0, 0.6, size=(n, 3))
    truth = market * np.exp(signal)
    truth /= truth.sum(axis=1, keepdims=True)
    model = market * np.exp(signal if informative else rng.normal(0, 0.6, size=(n, 3)))
    model /= model.sum(axis=1, keepdims=True)
    outcome = np.array([rng.choice(3, p=t) for t in truth])
    weeks = np.repeat(np.arange(n_weeks), 10)
    frame = pd.DataFrame(
        {
            "season": 2018 + weeks // 20,
            "league": "E0",
            "snapshot": pd.Timestamp("2018-08-06") + pd.to_timedelta(weeks * 7, unit="D"),
            "outcome": outcome,
        }
    )
    return frame, market, model


@needs_blend
def test_fit_beta_recovers_informative_model():
    frame, market, model = _simulated()
    beta = fit_beta(market, model, frame["outcome"].to_numpy())
    assert 0.6 < beta < 1.4


@needs_blend
def test_stack_test_passes_informative_and_fails_noise():
    seasons = [2020, 2021, 2022, 2023]
    frame, market, model = _simulated(informative=True)
    good = stack_test(frame, market, model, eval_seasons=seasons, n_boot=200)
    assert good.delta > 0.005 and good.criteria["beta interval above 0"]

    frame, market, model = _simulated(informative=False, seed=1)
    bad = stack_test(frame, market, model, eval_seasons=seasons, n_boot=200)
    assert not bad.passed
    assert bad.beta_pooled < 0.2
