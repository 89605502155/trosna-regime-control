import numpy as np
import pytest

from trosna_regime_control import aitchison_distance, allocate, clr, clr_inv, perturb
from trosna_regime_control.simplex import closure, eg_step, plan

RNG = np.random.default_rng(1)


def random_shares(n: int = 4) -> np.ndarray:
    return closure(RNG.uniform(0.05, 1.0, size=n))


def test_clr_sums_to_zero_and_inverts() -> None:
    w = random_shares(5)
    assert abs(clr(w).sum()) < 1e-12
    assert np.allclose(clr_inv(clr(w)), w)


def test_perturbation_is_addition_in_clr() -> None:
    w, p = random_shares(), random_shares()
    assert np.allclose(clr(perturb(w, p)), clr(w) + clr(p))


def test_identity_element_is_uniform() -> None:
    w = random_shares()
    assert np.allclose(perturb(w, np.full(4, 0.25)), w)


def test_aitchison_distance_examples_from_document() -> None:
    assert float(aitchison_distance([0.01, 0.99], [0.02, 0.98])) == pytest.approx(0.497, abs=1e-3)
    assert float(aitchison_distance([0.40, 0.60], [0.41, 0.59])) == pytest.approx(0.029, abs=1e-3)


def test_distance_does_not_depend_on_total_memory() -> None:
    m1 = np.array([2.0, 3.0, 5.0])
    m2 = np.array([2.5, 2.5, 5.0])
    assert float(aitchison_distance(m1, m2)) == pytest.approx(
        float(aitchison_distance(10 * m1, 10 * m2))
    )


def test_eg_step_is_multiplicative_weights() -> None:
    w = random_shares()
    g = RNG.normal(size=4)
    assert np.allclose(eg_step(w, g, 0.3), closure(w * np.exp(0.3 * g)))


def test_allocate_without_caps() -> None:
    alloc = allocate(100.0, [0.5, 0.3, 0.2], m_min=[10.0, 10.0, 20.0])
    assert np.allclose(alloc.memory, [40.0, 28.0, 32.0])
    assert alloc.unallocated == 0.0


def test_allocate_with_cap_redistributes_excess() -> None:
    alloc = allocate(100.0, [0.6, 0.2, 0.2], [10.0, 10.0, 10.0], [30.0, 80.0, 80.0])
    assert alloc.memory[0] == pytest.approx(30.0)
    assert alloc.memory.sum() == pytest.approx(100.0)
    assert alloc.memory[1] == pytest.approx(alloc.memory[2])


def test_allocate_reports_unallocated_when_all_capped() -> None:
    alloc = allocate(100.0, [0.5, 0.5], [10.0, 10.0], [30.0, 30.0])
    assert np.allclose(alloc.memory, [30.0, 30.0])
    assert alloc.unallocated == pytest.approx(40.0)


def test_allocate_validates_bounds() -> None:
    with pytest.raises(ValueError, match="minimum"):
        allocate(100.0, [0.5, 0.5], [40.0, 10.0], [30.0, 30.0])
    with pytest.raises(ValueError, match="available"):
        allocate(10.0, [0.5, 0.5], [10.0, 10.0])


def test_rejects_non_positive_shares() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        clr([0.5, 0.5, 0.0])


def test_dead_band_planner() -> None:
    small = plan([0.5, 0.5], [0.501, 0.499], delta=0.05)
    assert not small.reallocate
    assert np.allclose(small.shares, [0.5, 0.5])
    big = plan([0.5, 0.5], [0.7, 0.3], delta=0.05)
    assert big.reallocate
    assert np.allclose(big.shares, [0.7, 0.3])
    with pytest.raises(ValueError, match="non-negative"):
        plan([0.5, 0.5], [0.6, 0.4], delta=-1.0)
