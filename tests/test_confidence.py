"""Baseline confidence: the bounds widen with the level, and they cover the mean across seeds.

The coverage test uses 300 seeds with fixed values, so it is reproducible. It checks
that the one-sided bounds of the naive rate hold for about ``confidence`` of the runs.
The seeds were not used to choose any default.
"""

import math

import numpy as np
import pytest

from trosna_regime_control import (
    ControllerConfig,
    FixedInterval,
    ForgettingRate,
    RegimeController,
    StepIntensity,
    simulate,
)
from trosna_regime_control.intensity import expected_naive_rate

SEEDS = range(300)
RATE = 0.05
DT = 5.0
CHECKS = 400


def _bits(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = -math.expm1(-RATE * DT)
    return (rng.random(CHECKS) < p).astype(float)


def test_bounds_widen_with_confidence():
    bits = _bits(1)
    levels = [0.8, 0.9, 0.95, 0.99]
    ests = [ForgettingRate(tau=1e12, confidence=level) for level in levels]
    for est in ests:
        for bit in bits:
            est.record_check(float(bit), DT)
    ups = [est.upper_naive() for est in ests]
    lows = [est.lower_naive() for est in ests]
    assert ups == sorted(ups)
    assert lows == sorted(lows, reverse=True)
    naive = ests[0].naive
    assert all(low <= naive <= up for low, up in zip(lows, ups, strict=True))


@pytest.mark.parametrize("seed", [3, 11, 23, 42])
def test_controller_baselines_follow_confidence(seed):
    trace = simulate(FixedInterval(DT), StepIntensity(RATE), horizon_s=3000.0, seed=seed)
    low = RegimeController(confidence=0.80)
    high = RegimeController(confidence=0.99)
    low.start(0.0)
    high.start(0.0)
    for t, b in zip(trace.t, trace.decision, strict=True):
        low.record_decision(float(t), bool(b))
        high.record_decision(float(t), bool(b))
    assert low.est.upper_naive() <= high.est.upper_naive()
    assert low.est.lower_naive() >= high.est.lower_naive()


def test_z_replaces_confidence():
    ctrl = RegimeController(z=1.0)
    assert ctrl.config.z == 1.0
    assert ctrl.config.z_effective == 1.0
    assert ctrl.est.z == 1.0
    ctrl = RegimeController(ControllerConfig(z=1.0), confidence=0.99)
    assert ctrl.config.z is None
    assert ctrl.est.z == pytest.approx(ForgettingRate(1.0, confidence=0.99).z)


def test_confidence_and_z_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        RegimeController(confidence=0.9, z=1.0)


def test_coverage_of_one_sided_bounds_across_seeds():
    target = float(expected_naive_rate(RATE, DT))
    upper_hits = 0
    lower_hits = 0
    for seed in SEEDS:
        est = ForgettingRate(tau=1e12, confidence=0.95)
        for bit in _bits(seed):
            est.record_check(float(bit), DT)
        upper_hits += est.upper_naive() >= target
        lower_hits += est.lower_naive() <= target
    n = len(SEEDS)
    # Nominal 0.95; the fixed seeds give about 0.95 ± 0.013 (one standard error).
    assert upper_hits / n >= 0.90
    assert lower_hits / n >= 0.90


@pytest.mark.parametrize("seed", [5, 13])
def test_state_round_trip_keeps_confidence(seed):
    trace = simulate(FixedInterval(DT), StepIntensity(RATE), horizon_s=2000.0, seed=seed)
    ctrl = RegimeController(confidence=0.99)
    ctrl.start(0.0)
    for t, b in zip(trace.t, trace.decision, strict=True):
        ctrl.record_decision(float(t), bool(b))
    restored = RegimeController.from_json(ctrl.to_json())
    assert restored.config.confidence == 0.99
    assert restored.est.z == ctrl.est.z
    assert restored.est.upper_naive() == ctrl.est.upper_naive()


def test_version_1_state_is_refused():
    ctrl = RegimeController()
    ctrl.start(0.0)
    data = ctrl.to_dict()
    data["version"] = 1
    with pytest.raises(ValueError, match="unsupported state version"):
        RegimeController.from_dict(data)
