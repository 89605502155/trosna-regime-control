import math

import numpy as np
import pytest

from trosna_regime_control import (
    Cusum,
    ShiryaevRoberts,
    TwoSided,
    cusum_threshold_for_arl,
    lr_factor,
)
from trosna_regime_control.detectors import (
    cusum_path,
    expected_factor,
    first_passage_time,
    kl_per_check,
    kl_rate_poisson,
    log_lr_path,
    lorden_delay_bound,
    run_lengths,
    sr_path,
    sr_time_path,
)

LAM0, KAPPA = 0.05, 2.0


def test_factor_is_martingale_under_h0() -> None:
    dts = np.linspace(0.5, 20, 50)
    assert np.allclose(expected_factor(LAM0, dts, LAM0, KAPPA), 1.0, atol=1e-12)


def test_supermartingale_for_smaller_rates() -> None:
    dts = np.linspace(0.5, 20, 50)
    for lam in (0.01, 0.03, 0.049):
        assert np.all(expected_factor(lam, dts, LAM0, KAPPA) <= 1.0 + 1e-12)


def test_paths_match_stepwise_definitions() -> None:
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, size=(3, 200))
    dt = rng.uniform(1, 10, size=(3, 200))
    f = lr_factor(b, dt, LAM0, KAPPA)
    s_loop = np.zeros(3)
    r_loop = np.zeros(3)
    q_loop = np.zeros(3)
    s_path, r_path, q_path = cusum_path(f), sr_path(f), sr_time_path(f, dt)
    for k in range(200):
        s_loop = np.maximum(0.0, s_loop + np.log(f[:, k]))
        r_loop = (1.0 + r_loop) * f[:, k]
        q_loop = (dt[:, k] + q_loop) * f[:, k]
        assert np.allclose(s_path[:, k], s_loop)
        assert np.allclose(r_path[:, k], r_loop)
        assert np.allclose(q_path[:, k], q_loop)
    assert np.allclose(log_lr_path(f)[:, -1], np.log(f).sum(axis=1))


def test_cusum_never_exceeds_sr_with_a_equal_exp_h() -> None:
    rng = np.random.default_rng(4)
    b = rng.integers(0, 2, size=(5, 500))
    dt = rng.uniform(1, 10, size=(5, 500))
    f = lr_factor(b, dt, LAM0, KAPPA)
    assert np.all(np.exp(cusum_path(f)) <= np.maximum(1.0, sr_path(f)) * (1 + 1e-12))


def test_ville_guarantee_with_adaptive_intervals() -> None:
    alpha = 0.05
    rng = np.random.default_rng(5)
    dts = rng.uniform(1.0, 10.0, size=(3000, 400))
    bits = (rng.random(dts.shape) < -np.expm1(-LAM0 * dts)).astype(float)
    f = lr_factor(bits, dts, LAM0, KAPPA)
    crossed = (log_lr_path(f) >= np.log(1 / alpha)).any(axis=1).mean()
    assert crossed <= alpha + 0.01


def test_run_length_is_at_least_threshold_under_h0() -> None:
    rl = run_lengths("sr", 100.0, LAM0, LAM0, KAPPA, dt=1.0, n_runs=400, seed=6)
    assert np.isfinite(rl).mean() > 0.99
    assert rl[np.isfinite(rl)].mean() >= 100.0


def test_fast_run_lengths_match_formula_order() -> None:
    rl = run_lengths("cusum", math.log(100.0), LAM0, LAM0, KAPPA, n_runs=400, seed=7)
    assert rl[np.isfinite(rl)].mean() >= 100.0 * 0.8


def test_information_bound_numbers_from_document() -> None:
    assert lorden_delay_bound(3600.0, kl_rate_poisson(0.05, 2.0)) == pytest.approx(424, abs=1)
    assert lorden_delay_bound(3600.0, kl_rate_poisson(0.05, 4.0)) == pytest.approx(64, abs=1)
    assert kl_per_check(1.0, LAM0, KAPPA) < kl_rate_poisson(LAM0, KAPPA)


def test_first_passage_time_in_seconds() -> None:
    path = np.array([[0.0, 1.0, 5.0, 2.0]])
    t = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert first_passage_time(path, 4.0, t)[0] == pytest.approx(3.0)
    assert math.isinf(first_passage_time(path, 9.0, t)[0])


def test_online_detectors_match_paths_and_reset() -> None:
    rng = np.random.default_rng(8)
    b = rng.integers(0, 2, size=300)
    dt = rng.uniform(1, 10, size=300)
    path = sr_time_path(lr_factor(b, dt, LAM0, KAPPA)[None, :], dt[None, :])[0]
    det = ShiryaevRoberts(LAM0, KAPPA, threshold=math.inf, time_weighted=True)
    for k in range(300):
        det.record_check(float(b[k]), float(dt[k]))
        assert det.stat == pytest.approx(path[k])
    small = ShiryaevRoberts(LAM0, KAPPA, threshold=1e-9)
    assert small.record_check(1.0, 1.0)
    assert small.stat == 0.0
    cus = Cusum(LAM0, KAPPA, h=1e-9)
    assert cus.record_check(1.0, 1.0)
    assert cus.stat == 0.0
    assert TwoSided(LAM0, threshold=1e9).record_check(0.0, 1.0) == 0


def test_two_sided_reports_a_rise() -> None:
    rise = TwoSided(LAM0, threshold=1e-9)
    assert rise.record_check(1.0, 1.0) == 1
    assert rise.up.stat == 0.0
    assert rise.down.stat == 0.0


def test_cusum_threshold_for_arl() -> None:
    assert cusum_threshold_for_arl(3600.0) == pytest.approx(math.log(3600.0))
    with pytest.raises(ValueError, match="exceed"):
        cusum_threshold_for_arl(1.0)


def test_detector_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ShiryaevRoberts(0.0, 2.0, 10.0)
    with pytest.raises(ValueError, match="kind"):
        run_lengths("bad", 1.0, LAM0, LAM0, KAPPA)
