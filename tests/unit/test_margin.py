import numpy as np
import pytest

from football_edge.markets.margin import overround, remove_margin

SPAIN_ENGLAND = np.array([[1.96, 3.80, 3.35]])  # from the betting site, 15 Nov 2026


def _written() -> bool:
    try:
        remove_margin(SPAIN_ENGLAND)
        overround(SPAIN_ENGLAND)
    except NotImplementedError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _written(), reason="margin functions not written yet (docs/A4_1_margin_spec.md)"
)


def test_overround_worked_example():
    assert overround(SPAIN_ENGLAND) == pytest.approx([0.0719], abs=1e-4)


def test_remove_margin_worked_example():
    np.testing.assert_allclose(remove_margin(SPAIN_ENGLAND), [[0.4760, 0.2455, 0.2785]], atol=1e-4)


def test_rows_sum_to_one_and_keep_order():
    odds = np.array([[1.17, 7.60, 12.0], [2.15, 3.65, 3.05], [1.01, 11.0, 58.0]])
    p = remove_margin(odds)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (np.argsort(p, axis=1) == np.argsort(-odds, axis=1)).all()  # shortest odds = likeliest


def test_fair_odds_have_no_margin():
    fair = np.array([[2.0, 4.0, 4.0]])  # 0.5 + 0.25 + 0.25 = 1
    assert overround(fair) == pytest.approx([0.0])
    np.testing.assert_allclose(remove_margin(fair), [[0.5, 0.25, 0.25]])


@pytest.mark.parametrize(
    "odds",
    [
        np.array([[2.0]]),  # one column: not a market
        np.array([[1.0, 3.0, 4.0]]),  # odds must be above 1
        np.array([[2.0, np.nan, 4.0]]),  # missing price
        np.array([2.0, 3.0, 4.0]),  # not 2-D
    ],
)
def test_bad_odds_rejected(odds):
    with pytest.raises(ValueError):
        remove_margin(odds)
    with pytest.raises(ValueError):
        overround(odds)


def test_inputs_not_modified():
    odds = SPAIN_ENGLAND.copy()
    remove_margin(odds)
    overround(odds)
    np.testing.assert_array_equal(odds, SPAIN_ENGLAND)


def test_two_way_market():
    # Over/under 2.5 at 1.90 / 1.90: 5.3% margin, fair 50/50.
    odds = np.array([[1.90, 1.90]])
    assert overround(odds) == pytest.approx([2 / 1.9 - 1])
    np.testing.assert_allclose(remove_margin(odds), [[0.5, 0.5]])
