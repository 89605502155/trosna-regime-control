"""Validation of one check: types, finite and non-negative times, increasing times."""

import math

import numpy as np
import pytest

from trosna_regime_control import RegimeController, validate_decision


@pytest.mark.parametrize(
    ("decision", "expected"),
    [(True, True), (False, False), (1, True), (0, False), (np.bool_(True), True)],
)
def test_accepted_decisions(decision, expected):
    t, bit = validate_decision(10, decision)
    assert (t, bit) == (10.0, expected)
    assert type(t) is float
    assert type(bit) is bool


@pytest.mark.parametrize("decision", [2, -1, 5])
def test_integer_other_than_0_or_1(decision):
    with pytest.raises(ValueError, match="0 or 1"):
        validate_decision(1.0, decision)


@pytest.mark.parametrize("decision", [0.5, 1.0, "1", None])
def test_decision_wrong_type(decision):
    with pytest.raises(TypeError):
        validate_decision(1.0, decision)


@pytest.mark.parametrize("t", ["1", None, True, 1 + 2j])
def test_time_wrong_type(t):
    with pytest.raises(TypeError):
        validate_decision(t, True)


@pytest.mark.parametrize("t", [math.nan, math.inf, -math.inf, -1.0])
def test_time_not_finite_or_negative(t):
    with pytest.raises(ValueError):
        validate_decision(t, True)


def test_time_must_increase():
    with pytest.raises(ValueError, match="time must increase"):
        validate_decision(5.0, False, last_t=10.0)
    with pytest.raises(ValueError, match="time must increase"):
        validate_decision(10.0, False, last_t=10.0)
    assert validate_decision(10.5, False, last_t=10.0) == (10.5, False)


@pytest.mark.parametrize("bad_t", [math.nan, -3.0, "x"])
def test_refused_check_leaves_state_unchanged(bad_t):
    ctrl = RegimeController()
    ctrl.start(0.0)
    before = ctrl.to_json()
    with pytest.raises((TypeError, ValueError)):
        ctrl.record_decision(bad_t, True)
    assert ctrl.to_json() == before


def test_bad_decision_leaves_state_unchanged():
    ctrl = RegimeController()
    ctrl.start(0.0)
    before = ctrl.to_json()
    with pytest.raises(TypeError):
        ctrl.record_decision(5.0, 0.5)
    assert ctrl.to_json() == before
