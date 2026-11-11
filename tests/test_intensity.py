import math

import pytest

from trosna_regime_control import ForgettingRate, naive_bias
from trosna_regime_control.intensity import expected_naive_rate, relative_spread


def test_naive_bias_values_from_document() -> None:
    assert naive_bias(0.05, 1.0) == pytest.approx(-0.025, abs=5e-4)
    assert naive_bias(0.05, 5.0) == pytest.approx(-0.115, abs=5e-4)
    assert naive_bias(0.05, 10.0) == pytest.approx(-0.213, abs=5e-4)


def test_expected_naive_rate_matches_bias() -> None:
    assert float(expected_naive_rate(0.05, 10.0)) == pytest.approx(0.03935, abs=1e-4)


def test_debiased_rate_is_exact_for_constant_expectation() -> None:
    lam, dt = 0.05, 5.0
    p = -math.expm1(-lam * dt)
    est = ForgettingRate(tau=600.0)
    for _ in range(400):
        est.record_check(p, dt)  # expectation of the decision bit, no randomness
    assert est.debiased == pytest.approx(lam, rel=1e-6)
    assert est.naive == pytest.approx(float(expected_naive_rate(lam, dt)), rel=1e-6)


def test_forgetting_follows_real_time() -> None:
    fast = ForgettingRate(tau=10.0)
    slow = ForgettingRate(tau=1000.0)
    for est in (fast, slow):
        est.record_check(1.0, 1.0)
        est.record_check(0.0, 100.0)
    assert fast.n_moves < slow.n_moves  # the old move is forgotten more quickly


def test_prior_and_empty_state() -> None:
    est = ForgettingRate(tau=600.0)
    assert math.isnan(est.naive)
    assert math.isnan(est.debiased)
    prior = ForgettingRate(tau=600.0, prior_rate=0.05, prior_time=300.0)
    assert prior.naive == pytest.approx(0.05)


def test_state_roundtrip() -> None:
    est = ForgettingRate(tau=600.0)
    est.record_check(1.0, 3.0)
    restored = ForgettingRate.from_dict(est.to_dict())
    assert restored.debiased == est.debiased


def test_validation() -> None:
    with pytest.raises(ValueError, match="tau"):
        ForgettingRate(tau=0.0)
    est = ForgettingRate(tau=10.0)
    with pytest.raises(ValueError, match="dt"):
        est.record_check(1.0, 0.0)
    with pytest.raises(ValueError, match="decision"):
        est.record_check(2.0, 1.0)


def test_relative_spread_example() -> None:
    assert relative_spread(0.05, 600.0) == pytest.approx(0.1291, abs=1e-3)
