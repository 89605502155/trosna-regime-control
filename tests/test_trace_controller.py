import json

import numpy as np
import pytest

from trosna_regime_control import (
    ControllerConfig,
    FixedInterval,
    IntervalBounds,
    LegacyHeuristic,
    RegimeController,
    StepIntensity,
    read_checks,
    replay,
    simulate,
)
from trosna_regime_control.trace import Trace


def test_read_checks_with_dt_column(tmp_path) -> None:
    path = tmp_path / "checks.csv"
    path.write_text(
        "t_unix,dt_s,b,s_sum,mode\n100.0,10.0,0,0,0\n101.0,1.0,1,1,0\n104.0,3.0,0,1,0\n",
        encoding="utf-8",
    )
    tr = read_checks(path)
    assert len(tr) == 3
    assert np.allclose(tr.t, [10.0, 11.0, 14.0])
    assert tr.decision.tolist() == [0, 1, 0]
    assert tr.legacy_mode is not None
    assert tr.s_sum is not None


def test_read_checks_computes_dt_without_column(tmp_path) -> None:
    path = tmp_path / "checks.csv"
    path.write_text("t_unix,b\n100.0,0\n101.0,1\n104.0,0\n", encoding="utf-8")
    tr = read_checks(path)
    assert tr.dt.tolist() == [1.0, 3.0]
    assert tr.decision.tolist() == [1, 0]


def test_read_checks_errors(tmp_path) -> None:
    empty = tmp_path / "e.csv"
    empty.write_text("t_unix,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_checks(empty)
    missing = tmp_path / "m.csv"
    missing.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        read_checks(missing)


def test_trace_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        Trace(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="0 or 1"):
        Trace(np.array([1.0]), np.array([1.0]), np.array([2]))


def test_controller_needs_start() -> None:
    with pytest.raises(RuntimeError, match="start"):
        RegimeController().record_decision(1.0, True)


def test_controller_stationary_uses_rule_a() -> None:
    ctrl = RegimeController(ControllerConfig(rule="A", p_star=0.3, warmup_s=1e9))
    assert ctrl.start(0.0) == pytest.approx(10.0)
    t = 0.0
    for _ in range(200):
        t += 5.0
        step = ctrl.record_decision(t, decision=(int(t) % 3 == 0))
    assert not step.nonstationary
    assert step.alarm is None
    assert step.next_dt == pytest.approx(min(max(0.3 / step.rate, 1.0), 10.0))


def test_controller_rule_b_runs() -> None:
    ctrl = RegimeController(ControllerConfig(rule="B", c1_over_c2=2.0, warmup_s=1e9))
    ctrl.start(0.0)
    step = ctrl.record_decision(4.0, decision=True)
    assert 1.0 <= step.next_dt <= 10.0


def test_controller_rejects_non_increasing_time() -> None:
    ctrl = RegimeController()
    ctrl.start(0.0)
    ctrl.record_decision(5.0, False)
    with pytest.raises(ValueError, match="increase"):
        ctrl.record_decision(5.0, False)


def test_controller_detects_a_rise_and_relaxes_interval() -> None:
    cfg = ControllerConfig(
        rule="A", arl_s=1e5, warmup_s=600.0, hold_s=120.0, tau_s=900.0, tau_fast_s=120.0
    )
    intensity = StepIntensity(0.02, 0.10, t_change=3000.0)
    tr = simulate(RegimeController(cfg), intensity, horizon_s=6000.0, seed=41)
    steps = replay(RegimeController(cfg), tr)
    rises = [k for k, s in enumerate(steps) if s.alarm == "rise"]
    assert rises, "a fivefold rise must be detected within the horizon"
    first = rises[0]
    assert tr.t[first] > 3000.0
    assert steps[first].nonstationary
    assert steps[first].next_dt == pytest.approx(1.0)


def test_controller_state_roundtrip_continues_identically() -> None:
    cfg = ControllerConfig(rule="A", warmup_s=60.0, arl_s=200.0)
    tr = simulate(RegimeController(cfg), StepIntensity(0.05), 2000.0, seed=42)
    half = len(tr) // 2
    ctrl = RegimeController(cfg)
    ctrl.start(0.0)
    for k in range(half):
        ctrl.record_decision(float(tr.t[k]), bool(tr.decision[k]))
    restored = RegimeController.from_json(ctrl.to_json())
    assert json.loads(restored.to_json()) == json.loads(ctrl.to_json())
    for k in range(half, len(tr)):
        a = restored.record_decision(float(tr.t[k]), bool(tr.decision[k]))
        b = ctrl.record_decision(float(tr.t[k]), bool(tr.decision[k]))
        assert a.next_dt == pytest.approx(b.next_dt)
        assert a.alarm == b.alarm
        assert a.nonstationary == b.nonstationary


def test_state_version_is_checked() -> None:
    ctrl = RegimeController()
    data = ctrl.to_dict()
    data["version"] = 99
    with pytest.raises(ValueError, match="version"):
        RegimeController.from_dict(data)


def test_bounds_come_from_config() -> None:
    cfg = ControllerConfig(bounds=IntervalBounds(lo=2.0, hi=4.0), warmup_s=1e9)
    ctrl = RegimeController(cfg)
    assert ctrl.start(0.0) == pytest.approx(4.0)
    assert ctrl.record_decision(4.0, True).next_dt >= 2.0


def test_legacy_and_fixed_schedulers_in_simulation() -> None:
    tr = simulate(LegacyHeuristic(), StepIntensity(0.05), 3600.0, seed=43)
    assert len(tr) > 100
    tr_fixed = simulate(FixedInterval(4.0), StepIntensity(0.05), 400.0, seed=44)
    assert np.allclose(tr_fixed.dt, 4.0)
    assert len(tr_fixed) == 100
