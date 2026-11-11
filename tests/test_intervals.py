import math

import numpy as np
import pytest

from trosna_regime_control import IntervalBounds, exact_interval, rule_a, rule_b, rule_c
from trosna_regime_control.intervals import cost_rate, steady_state_rule_a


def test_rule_values_and_clipping() -> None:
    # exact (default): -ln(1 - 0.3) / 0.05 = 7.1335; linear: 0.3 / 0.05 = 6
    assert float(rule_a(0.05, 0.3)) == pytest.approx(-math.log(0.7) / 0.05)
    assert float(rule_a(0.05, 0.3)) == pytest.approx(7.1335, abs=1e-4)
    assert float(rule_a(0.05, 0.3, formula="linear")) == pytest.approx(6.0)
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
    # inside the bounds: exact -ln(0.7)/0.1 = 3.567, linear 0.3/0.1 = 3
    assert float(rule_a(0.1, 0.3, bounds)) == pytest.approx(-math.log(0.7) / 0.1)
    assert float(rule_a(0.1, 0.3, bounds, formula="linear")) == pytest.approx(3.0)


def test_exact_formula_is_probability_inversion() -> None:
    # P(N(dt) >= 1) = 1 - exp(-rate * dt) must equal p* at the exact interval.
    bounds = IntervalBounds(lo=1e-9, hi=1e9)
    for rate in (0.01, 0.05, 0.2):
        for p in (0.05, 0.3, 0.7):
            dt = float(rule_a(rate, p, bounds))
            assert 1.0 - math.exp(-rate * dt) == pytest.approx(p, abs=1e-12)


def test_exact_is_about_19_percent_longer_at_p_03() -> None:
    bounds = IntervalBounds(lo=1e-9, hi=1e9)
    exact = float(rule_a(0.05, 0.3, bounds))
    linear = float(rule_a(0.05, 0.3, bounds, formula="linear"))
    assert exact / linear == pytest.approx(1.1889, abs=1e-4)
    assert 0.18 < exact / linear - 1.0 < 0.20


def test_exact_and_linear_converge_for_small_p() -> None:
    # -ln(1 - p) / p = 1 + p/2 + p^2/3 + ..., so the relative gap is about p/2.
    bounds = IntervalBounds(lo=1e-9, hi=1e9)
    for p in (1e-4, 1e-3, 1e-2):
        ratio = float(rule_a(0.05, p, bounds)) / float(rule_a(0.05, p, bounds, formula="linear"))
        assert ratio - 1.0 == pytest.approx(p / 2.0, abs=p * p)


def test_linear_never_exceeds_exact() -> None:
    bounds = IntervalBounds(lo=1e-9, hi=1e9)
    for p in np.linspace(0.01, 0.99, 50):
        exact = float(rule_a(0.05, float(p), bounds))
        linear = float(rule_a(0.05, float(p), bounds, formula="linear"))
        assert linear <= exact


def test_rule_a_scale_multiplies_the_interval() -> None:
    bounds = IntervalBounds(lo=1e-9, hi=1e9)
    base = float(rule_a(0.05, 0.3, bounds))
    assert float(rule_a(0.05, 0.3, bounds, scale=2.5)) == pytest.approx(2.5 * base)
    assert float(rule_a(0.05, 0.3, bounds, formula="linear", scale=0.5)) == pytest.approx(0.5 * 6.0)


def test_rule_a_validation() -> None:
    with pytest.raises(ValueError, match="p_star"):
        rule_a(0.05, 0.0)
    with pytest.raises(ValueError, match="p_star"):
        rule_a(0.05, 1.0)
    with pytest.raises(ValueError, match="p_star"):
        rule_a(0.05, np.nan)
    with pytest.raises(TypeError, match="p_star"):
        rule_a(0.05, "0.3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="formula"):
        rule_a(0.05, 0.3, formula="log")
    with pytest.raises(TypeError, match="formula"):
        rule_a(0.05, 0.3, formula=1)  # type: ignore[arg-type]
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="scale"):
            rule_a(0.05, 0.3, scale=bad)
    with pytest.raises(TypeError, match="scale"):
        rule_a(0.05, 0.3, scale="2")  # type: ignore[arg-type]


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
