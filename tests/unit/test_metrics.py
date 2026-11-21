import numpy as np
import pytest

from football_edge.evaluation.metrics import brier_score, log_loss


def _log_loss_written() -> bool:
    try:
        log_loss(np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([0]))
    except NotImplementedError:
        return False
    return True


needs_log_loss = pytest.mark.skipif(
    not _log_loss_written(), reason="log_loss not written yet (docs/A1_4_log_loss_spec.md)"
)


def test_brier_uniform_is_two_thirds():
    p = np.full((4, 3), 1 / 3)
    assert brier_score(p, np.array([0, 1, 2, 0])) == pytest.approx(2 / 3)


def test_brier_perfect_is_zero():
    assert brier_score(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0)


@needs_log_loss
def test_log_loss_uniform_is_ln3():
    p = np.full((5, 3), 1 / 3)
    assert log_loss(p, np.array([0, 1, 2, 0, 1])) == pytest.approx(np.log(3))


@needs_log_loss
def test_log_loss_worked_example():
    # Spec section 3: two matches, home win at 0.5 then draw at 0.3.
    p = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    expected = -(np.log(0.5) + np.log(0.3)) / 2
    assert log_loss(p, np.array([0, 1])) == pytest.approx(expected)


@needs_log_loss
def test_log_loss_perfect_is_near_zero():
    assert log_loss(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0, abs=1e-12)


@needs_log_loss
def test_log_loss_zero_probability_is_finite():
    p = np.array([[1.0, 0.0, 0.0]])
    value = log_loss(p, np.array([1]))
    assert np.isfinite(value) and value > 30  # -ln(1e-15) is about 34.5


@needs_log_loss
@pytest.mark.parametrize(
    "probs, outcomes",
    [
        (np.full((2, 3), 1 / 3), np.array([0])),  # length mismatch
        (np.ones((2, 1)), np.array([0, 0])),  # one column: not a forecast
        (np.array([[0.5, 0.5, 0.5]]), np.array([0])),  # row sums to 1.5
        (np.array([[1.2, -0.1, -0.1]]), np.array([0])),  # outside [0, 1]
        (np.full((1, 3), 1 / 3), np.array([3])),  # outcome not 0/1/2
    ],
)
def test_log_loss_rejects_bad_input(probs, outcomes):
    with pytest.raises(ValueError):
        log_loss(probs, outcomes)


@needs_log_loss
def test_log_loss_does_not_modify_inputs():
    p = np.array([[1.0, 0.0, 0.0]])
    y = np.array([0])
    log_loss(p, y)
    assert p.tolist() == [[1.0, 0.0, 0.0]]


@needs_log_loss
def test_log_loss_works_for_two_outcomes():
    # Over/under: over 2.5 at 0.6 happened, then under at 0.3 happened.
    p = np.array([[0.6, 0.4], [0.7, 0.3]])
    assert log_loss(p, np.array([0, 1])) == pytest.approx(-(np.log(0.6) + np.log(0.3)) / 2)
