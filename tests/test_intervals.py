import numpy as np
import pytest

from trosna_regime_control import IntervalBounds, exact_interval, rule_a, rule_b, rule_c
from trosna_regime_control.intervals import cost_rate, steady_state_rule_a


def test_rule_values_and_clipping() -> None:
    assert float(rule_a(0.05, 0.3)) == pytest.approx(6.0)
    assert float(rule_a(1e-6, 0.3)) == pytest.approx(10.0)
    assert float(rule_a(np.nan, 0.3)) == pytest.approx(10.0)
    assert float(rule_a(np.nan, 0.3, fallback=2.0)) == pytest.approx(2.0)
    assert float(rule_a(0.0, 0.3)) == pytest.approx(10.0)
    assert float(rule_b(0.05, 2.0)) == pytest.approx(np.sqrt(80.0))
    assert float(rule_c(0.1, 0.3)) == pytest.approx(3.0)


def test_rules_respect_custom_bounds() -> None:
    bounds = IntervalBounds(lo=2.0, hi=5.0)
    assert float(rule_a(1.0, 0.3, bounds)) == pytest.approx(2.0)
    assert float(rule_a(0.01, 0.3, bounds)) == pytest.approx(5.0)


def test_bounds_validation() -> None:
    with pytest.raises(ValueError, match="lo <= hi"):
        IntervalBounds(lo=5.0, hi=1.0)
    with pytest.raises(ValueError, match="C1/C2"):
        rule_b(0.05, 0.0)


def test_rule_b_is_cost_minimum() -> None:
    lam, c1, c2 = 0.05, 2.0, 1.0
    grid = np.linspace(1, 30, 20001)
    best = grid[np.argmin(cost_rate(grid, lam, c1, c2))]
    assert best == pytest.approx(float(rule_b(lam, c1 / c2, IntervalBounds(1, 1000))), abs=0.01)


def test_exact_interval_and_rule_b_numbers_from_document() -> None:
    assert exact_interval(0.05, 2.0) == pytest.approx(8.32, abs=0.01)
    assert np.sqrt(2 * 2.0 / 0.05) == pytest.approx(8.94, abs=0.01)
    with pytest.raises(ValueError, match="positive"):
        exact_interval(0.0, 2.0)


def test_rule_a_steady_state_share() -> None:
    _, share_new = steady_state_rule_a(0.05, 0.3, "p_over_rate")
    _, share_log = steady_state_rule_a(0.05, 0.3, "log")
    assert share_new == pytest.approx(0.300, abs=1e-6)
    assert share_log == pytest.approx(0.357, abs=1e-3)
    with pytest.raises(ValueError, match="variant"):
        steady_state_rule_a(0.05, 0.3, "other")
