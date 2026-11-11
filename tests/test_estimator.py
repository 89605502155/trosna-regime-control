"""Rate estimator choice (naive or debiased) and the four combinations of rule A."""

import json
import math

import pytest

from trosna_regime_control import (
    ControllerConfig,
    ForgettingRate,
    IntervalBounds,
    RegimeController,
    StepIntensity,
    UniformIntervals,
    simulate,
)
from trosna_regime_control.cli import main
from trosna_regime_control.intensity import check_estimator
from trosna_regime_control.intervals import steady_state_rule_a, steady_state_share

COMBINATIONS = [
    ("exact", "naive"),
    ("exact", "debiased"),
    ("linear", "naive"),
    ("linear", "debiased"),
]
WIDE = IntervalBounds(lo=0.01, hi=1000.0)


def test_check_estimator_validation() -> None:
    check_estimator("naive")
    check_estimator("debiased")
    with pytest.raises(ValueError, match="estimator"):
        check_estimator("unbiased")
    with pytest.raises(TypeError, match="string"):
        check_estimator(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="estimator"):
        ControllerConfig(estimator="mle")  # type: ignore[arg-type]


def test_default_estimator_is_debiased_with_exact_formula() -> None:
    cfg = ControllerConfig()
    assert cfg.estimator == "debiased"
    assert cfg.formula == "exact"


def test_forgetting_rate_estimate_selects_the_two_estimates() -> None:
    est = ForgettingRate(tau=600.0)
    est.record_check(1.0, 5.0)
    est.record_check(0.0, 3.0)
    assert est.estimate("naive") == est.naive
    assert est.estimate("debiased") == est.debiased
    assert est.estimate() == est.debiased
    assert math.isnan(ForgettingRate(tau=10.0).estimate("naive"))
    with pytest.raises(ValueError, match="estimator"):
        est.estimate("other")


def test_controller_rate_follows_config_estimator() -> None:
    naive = RegimeController(ControllerConfig(estimator="naive", warmup_s=1e9))
    debiased = RegimeController(ControllerConfig(estimator="debiased", warmup_s=1e9))
    for ctrl in (naive, debiased):
        ctrl.start(0.0)
        for k in range(1, 40):
            ctrl.record_decision(5.0 * k, decision=(k % 3 == 0))
    assert naive.rate == naive.est.naive
    assert debiased.rate == debiased.est.debiased
    assert naive.rate != pytest.approx(debiased.rate)


@pytest.mark.parametrize(("formula", "estimator"), COMBINATIONS)
def test_every_combination_is_selectable_and_runs(formula: str, estimator: str) -> None:
    cfg = ControllerConfig(formula=formula, estimator=estimator, warmup_s=1e9)  # type: ignore[arg-type]
    ctrl = RegimeController(cfg)
    assert ctrl.config.formula == formula
    assert ctrl.config.estimator == estimator
    ctrl.start(0.0)
    step = ctrl.record_decision(10.0, decision=True)
    assert 1.0 <= step.next_dt <= 10.0


@pytest.mark.parametrize(("formula", "estimator"), COMBINATIONS)
def test_simulated_share_matches_analytic_steady_state(formula: str, estimator: str) -> None:
    # Three seeds; the share after burn-in must match the analytic value and λΔt
    # the analytic product. Tolerances are about four standard errors.
    rate, p_star = 0.05, 0.3
    share_a, lam_dt_a = steady_state_share(p_star, formula, estimator)  # type: ignore[arg-type]
    for seed in (5, 6, 7):
        cfg = ControllerConfig(
            formula=formula,  # type: ignore[arg-type]
            estimator=estimator,  # type: ignore[arg-type]
            p_star=p_star,
            bounds=WIDE,
            warmup_s=math.inf,
        )
        tr = simulate(RegimeController(cfg), StepIntensity(rate), 60_000.0, seed=seed)
        post = tr.t >= 9000.0
        assert float(tr.decision[post].mean()) == pytest.approx(share_a, abs=0.02)
        assert rate * float(tr.dt[post].mean()) == pytest.approx(lam_dt_a, rel=0.05)


def test_exact_debiased_and_linear_naive_share_the_target() -> None:
    # The two consistent pairs give the target share and the same λΔt.
    for p in (0.05, 0.3, 0.5):
        a = steady_state_share(p, "exact", "debiased")
        b = steady_state_share(p, "linear", "naive")
        assert a[0] == pytest.approx(p, abs=1e-12)
        assert b[0] == pytest.approx(p, abs=1e-12)
        assert a[1] == pytest.approx(-math.log1p(-p), abs=1e-12)
        assert b[1] == pytest.approx(a[1], abs=1e-12)


def test_steady_state_closed_forms_for_the_other_pairs() -> None:
    share, lam_dt = steady_state_share(0.3, "exact", "naive")
    assert share == pytest.approx(-math.log(0.7), abs=1e-12)
    assert lam_dt == pytest.approx(-math.log(1.0 - share), abs=1e-12)
    share, lam_dt = steady_state_share(0.3, "linear", "debiased")
    assert share == pytest.approx(1.0 - math.exp(-0.3), abs=1e-12)
    assert lam_dt == pytest.approx(0.3, abs=1e-12)


def test_closed_forms_agree_with_the_iteration() -> None:
    _, share_lin = steady_state_rule_a(0.05, 0.3, "p_over_rate")
    _, share_exact = steady_state_rule_a(0.05, 0.3, "log")
    assert steady_state_share(0.3, "linear", "naive")[0] == pytest.approx(share_lin, abs=1e-6)
    assert steady_state_share(0.3, "exact", "naive")[0] == pytest.approx(share_exact, abs=1e-6)


def test_steady_state_scale_and_validation() -> None:
    share, _ = steady_state_share(0.3, "linear", "debiased", scale=2.0)
    assert share == pytest.approx(1.0 - math.exp(-0.6), abs=1e-12)
    with pytest.raises(ValueError, match="no steady state"):
        steady_state_share(0.5, "exact", "naive", scale=2.0)
    with pytest.raises(ValueError, match="estimator"):
        steady_state_share(0.3, "exact", "other")
    with pytest.raises(ValueError, match="p_star"):
        steady_state_share(1.0, "exact", "naive")


def test_state_roundtrip_keeps_estimator() -> None:
    cfg = ControllerConfig(estimator="naive", warmup_s=1e9)
    ctrl = RegimeController(cfg)
    ctrl.start(0.0)
    ctrl.record_decision(10.0, decision=True)
    restored = RegimeController.from_json(ctrl.to_json())
    assert restored.config.estimator == "naive"
    assert restored.rate == ctrl.rate


def test_state_without_estimator_field_restores_debiased() -> None:
    # A state saved before the field existed used the debiased rate.
    ctrl = RegimeController(ControllerConfig(warmup_s=1e9))
    ctrl.start(0.0)
    ctrl.record_decision(10.0, decision=True)
    data = ctrl.to_dict()
    del data["config"]["estimator"]
    restored = RegimeController.from_dict(data)
    assert restored.config.estimator == "debiased"
    assert restored.rate == pytest.approx(ctrl.rate)


def test_cli_estimator_option(tmp_path, capsys) -> None:
    tr = simulate(
        UniformIntervals(1.0, 10.0, seed=3),
        StepIntensity(0.05),
        3000.0,
        seed=4,
    )
    path = tmp_path / "checks.csv"
    lines = ["t_unix,dt_s,b"] + [
        f"{1_763_200_000.0 + t:.3f},{dt:.3f},{int(b)}"
        for t, dt, b in zip(tr.t, tr.dt, tr.decision, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["analyze", str(path), "--json", "--estimator", "naive"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_checks"] > 100
    assert main(["analyze", str(path), "--json"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["analyze", str(path), "--estimator", "unbiased"])
    capsys.readouterr()
