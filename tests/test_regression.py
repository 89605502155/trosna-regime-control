import numpy as np
import pytest

from trosna_regime_control import UniformIntervals, fit_cloglog, ones_share_by_interval, simulate
from trosna_regime_control.synthetic import StepIntensity


def test_fit_accepts_correct_model() -> None:
    tr = simulate(UniformIntervals(1.0, 10.0, seed=3), StepIntensity(0.05), 9000.0, seed=31)
    fit = fit_cloglog(tr.decision, tr.dt, burn_in=5)
    assert fit.converged
    assert fit.slope_ln_dt.estimate == pytest.approx(1.0, abs=0.25)
    assert fit.rate_hat == pytest.approx(0.05, rel=0.2)
    assert fit.model_consistent


def test_fit_detects_carryover() -> None:
    tr = simulate(
        UniformIntervals(1.0, 10.0, seed=4), StepIntensity(0.2), 20_000.0, seed=32, carryover=0.5
    )
    fit = fit_cloglog(tr.decision, tr.dt, burn_in=5)
    assert fit.coef_prev_bit.ci95[0] > 0.0
    assert not fit.model_consistent


def test_fit_recovers_coefficients_on_synthetic_data() -> None:
    rng = np.random.default_rng(33)
    dts = rng.uniform(1.0, 10.0, 6000)
    bits = (rng.random(dts.shape) < -np.expm1(-0.05 * dts)).astype(int)
    fit = fit_cloglog(bits, dts)
    assert fit.slope_ln_dt.contains(1.0)
    assert fit.coef_prev_bit.contains(0.0)
    assert fit.rate_hat == pytest.approx(0.05, rel=0.1)


def test_fit_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="too few"):
        fit_cloglog([0, 1, 0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        fit_cloglog([0, 1], [1.0])


def test_share_by_interval_groups() -> None:
    out = ones_share_by_interval([0, 1, 1, 0, 1], [1.0, 1.0, 2.0, 2.0, 2.0])
    assert out.dt.tolist() == [1.0, 2.0]
    assert out.share.tolist() == pytest.approx([0.5, 2.0 / 3.0])
    assert out.count.tolist() == [2, 3]
