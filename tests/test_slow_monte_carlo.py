"""Monte Carlo check of the ARL guarantee. Deselected by default: run with -m slow."""

import numpy as np
import pytest

from trosna_regime_control.detectors import run_lengths

pytestmark = pytest.mark.slow


def test_sr_arl_guarantee_holds_with_many_runs() -> None:
    rl = run_lengths("sr", 300.0, 0.05, 0.05, 2.0, dt=1.0, n_runs=4000, seed=61)
    assert np.isfinite(rl).all()
    assert rl.mean() >= 300.0 * 0.97
