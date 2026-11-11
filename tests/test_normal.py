"""Standard normal quantile (AS 241) and the confidence-to-z conversion."""

import itertools
import math

import numpy as np
import pytest

from trosna_regime_control import norm_ppf, z_for_confidence
from trosna_regime_control.normal import baseline_z, check_z

# Reference values of Φ⁻¹ to 15 digits (standard tables and high-precision evaluation).
KNOWN = [
    (0.95, 1.6448536269514722),
    (0.975, 1.959963984540054),
    (0.99, 2.3263478740408408),
    (0.999, 3.0902323061678132),
    (1e-6, -4.753424308822899),
]


@pytest.mark.parametrize(("p", "value"), KNOWN)
def test_known_values(p, value):
    assert norm_ppf(p) == pytest.approx(value, abs=1e-12)


def test_median_and_symmetry():
    assert norm_ppf(0.5) == 0.0
    # k / 1024 is exact in binary, so 1 − p is exact too (no rounding of the mirror point).
    for k in range(1, 512):
        p = k / 1024.0
        assert norm_ppf(1.0 - p) == pytest.approx(-norm_ppf(p), abs=1e-13)


def test_round_trip_through_erfc():
    # Φ(x) = ½ erfc(−x/√2) keeps the relative accuracy in both tails.
    for p in np.geomspace(1e-9, 0.5, 60).tolist() + (1.0 - np.geomspace(1e-9, 0.5, 60)).tolist():
        x = norm_ppf(p)
        back = 0.5 * math.erfc(-x / math.sqrt(2.0))
        assert abs(back - p) <= 1e-12 * max(p, 1.0 - p)


def test_monotone():
    values = [norm_ppf(p) for p in np.linspace(0.001, 0.999, 500)]
    assert all(b > a for a, b in itertools.pairwise(values))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_probability_out_of_range(bad):
    with pytest.raises(ValueError, match="strictly between"):
        norm_ppf(bad)


@pytest.mark.parametrize("bad", [True, "0.9", None])
def test_probability_wrong_type(bad):
    with pytest.raises(TypeError):
        norm_ppf(bad)


def test_confidence_to_z():
    assert z_for_confidence(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)
    assert z_for_confidence(0.5) == 0.0
    assert z_for_confidence(0.9) > 1.28
    assert z_for_confidence(0.9) < 1.2817


@pytest.mark.parametrize("bad", [0.49, 1.0, 1.2, -0.5, math.nan])
def test_confidence_out_of_range(bad):
    with pytest.raises(ValueError, match=r"\[0.5, 1\)"):
        z_for_confidence(bad)


def test_confidence_wrong_type():
    with pytest.raises(TypeError):
        z_for_confidence("0.95")


def test_check_z_and_baseline_z():
    assert check_z(2) == 2.0
    assert baseline_z(0.99) == pytest.approx(z_for_confidence(0.99))
    assert baseline_z(0.99, z=1.0) == 1.0  # z replaces the confidence
    with pytest.raises(ValueError, match="finite and non-negative"):
        check_z(-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        check_z(math.inf)
    with pytest.raises(TypeError):
        check_z(True)
