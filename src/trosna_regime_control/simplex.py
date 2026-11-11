"""Memory shares as points of the simplex: the Aitchison geometry (idea 1).

Shares ``w`` of a memory pool (all positive, summing to one) form a commutative
group under the perturbation operation

    (w ⊕ p)_i = w_i p_i / Σ_j w_j p_j,

with the neutral element ``(1/n, …, 1/n)``. Its logarithm is the centred
log-ratio (clr) transform and its exponential is the softmax. In clr
coordinates a control step needs no constraints, and the softmax always
returns valid shares. Because the group is commutative, the geometry is
ℝⁿ⁻¹ in disguise; the group language is a way to justify and write the method.

References: Aitchison (1986); Pawlowsky-Glahn and Egozcue (2001); Kivinen and
Warmuth (1997) for the exponentiated-gradient step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import FloatArray


def _positive(x: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        msg = "all components must be finite and strictly positive"
        raise ValueError(msg)
    return arr


def closure(x: npt.ArrayLike) -> FloatArray:
    """Scale positive components so that they sum to one (along the last axis).

    Parameters
    ----------
    x : array_like
        Positive weights, shape ``(..., n)``.

    Returns
    -------
    numpy.ndarray
        Shares of the same shape, summing to one along the last axis.

    Examples
    --------
    >>> closure([1.0, 3.0]).tolist()
    [0.25, 0.75]
    """
    arr = _positive(x)
    return np.asarray(arr / arr.sum(axis=-1, keepdims=True), dtype=np.float64)


def perturb(w: npt.ArrayLike, p: npt.ArrayLike) -> FloatArray:
    """Group operation: perturb shares ``w`` by shares ``p``.

    ``(w ⊕ p)_i = w_i p_i / Σ_j w_j p_j``. A perturbation by ``p`` that doubles
    the share of VM 1 relative to the others is ``p = (2, 1, …, 1)``, closed.

    Examples
    --------
    >>> perturb([0.5, 0.5], [1.0, 3.0]).tolist()
    [0.25, 0.75]
    """
    return closure(_positive(w) * _positive(p))


def clr(w: npt.ArrayLike) -> FloatArray:
    """Logarithm of the group: ``z_i = ln w_i − mean_j ln w_j``.

    The coordinates sum to zero, and ``clr(w ⊕ p) = clr(w) + clr(p)``.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.allclose(clr([0.25, 0.75]), [-0.5493061, 0.5493061]))
    True
    """
    lw = np.log(_positive(w))
    return np.asarray(lw - lw.mean(axis=-1, keepdims=True), dtype=np.float64)


def clr_inv(z: npt.ArrayLike) -> FloatArray:
    """Exponential of the group (softmax). Inverse of :func:`clr` on sum-zero vectors.

    Examples
    --------
    >>> clr_inv([-0.5493061443, 0.5493061443]).round(4).tolist()
    [0.25, 0.75]
    """
    arr = np.asarray(z, dtype=np.float64)
    shifted = arr - arr.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float64)


def aitchison_distance(w1: npt.ArrayLike, w2: npt.ArrayLike) -> FloatArray:
    """Aitchison distance ``d = ‖clr(w1) − clr(w2)‖``.

    It does not depend on the total memory of the host. It measures a relative
    change: for two VMs a share growing from 1 % to 2 % gives ``d ≈ 0.50``, and
    from 40 % to 41 % only ``d ≈ 0.03``. Calibrate the dead band δ on traces.

    Examples
    --------
    >>> round(float(aitchison_distance([0.01, 0.99], [0.02, 0.98])), 3)
    0.497
    """
    diff = clr(w1) - clr(w2)
    return np.asarray(np.linalg.norm(diff, axis=-1), dtype=np.float64)


def eg_step(w: npt.ArrayLike, g: npt.ArrayLike, eta: float) -> FloatArray:
    """Exponentiated-gradient step in clr coordinates: ``z ← z + η g``, then softmax.

    ``g`` is the ascent direction for each share, for example the normalised
    memory-pressure signal of each VM. The step equals the multiplicative
    update ``w_i ∝ w_i exp(η g_i)``.
    """
    return clr_inv(clr(w) + eta * np.asarray(g, dtype=np.float64))


@dataclass(frozen=True)
class Allocation:
    """Memory of every VM and the part of the pool that no VM could take.

    Attributes
    ----------
    memory : numpy.ndarray
        Memory per VM, MiB.
    unallocated : float
        Pool left over when every VM is at its maximum, MiB (zero otherwise).
    """

    memory: FloatArray
    unallocated: float


def allocate(
    total: float,
    shares: npt.ArrayLike,
    m_min: npt.ArrayLike,
    m_max: npt.ArrayLike | None = None,
    tol: float = 1e-9,
) -> Allocation:
    """Memory per VM from shares of the free pool, with minimum and maximum bounds.

    ``m_i = m_min_i + w_i P``, where ``P = total − Σ m_min``. A VM that reaches
    its maximum is fixed there, and the excess goes to the others in proportion
    to their shares. Bounds are the only constraints; the result never exceeds
    ``total`` and never falls below ``m_min``.

    Parameters
    ----------
    total : float
        Memory available to the VMs, MiB.
    shares : array_like
        Target shares of the free pool, positive.
    m_min : array_like
        Minimum memory per VM, MiB.
    m_max : array_like, optional
        Maximum memory per VM, MiB. Default: no maximum.
    tol : float
        Numerical tolerance, MiB.

    Examples
    --------
    >>> alloc = allocate(100.0, [0.6, 0.2, 0.2], [10.0, 10.0, 10.0], [30.0, 80.0, 80.0])
    >>> alloc.memory.round(1).tolist()
    [30.0, 35.0, 35.0]
    """
    w = closure(shares)
    lo = np.asarray(m_min, dtype=np.float64)
    n = w.shape[-1]
    if lo.shape != (n,):
        msg = "m_min must have one entry per VM"
        raise ValueError(msg)
    hi = np.full(n, np.inf) if m_max is None else np.asarray(m_max, dtype=np.float64)
    if hi.shape != (n,):
        msg = "m_max must have one entry per VM"
        raise ValueError(msg)
    if np.any(lo > hi):
        msg = "the minimum of a VM is above its maximum"
        raise ValueError(msg)
    pool = float(total) - float(lo.sum())
    if pool < -tol:
        msg = "the sum of minimums is above the available memory"
        raise ValueError(msg)

    mem = lo.copy()
    free = np.ones(n, dtype=bool)
    remaining = max(pool, 0.0)
    for _ in range(n + 1):
        if remaining <= tol or not free.any():
            break
        share = np.where(free, w, 0.0)
        share = share / share.sum()
        add = remaining * share
        room = hi - mem
        over = free & (add > room + tol)
        if not over.any():
            mem = mem + add
            remaining = 0.0
            break
        mem[over] = hi[over]
        remaining -= float(room[over].sum())
        free &= ~over
    return Allocation(memory=np.asarray(mem, dtype=np.float64), unallocated=max(remaining, 0.0))


@dataclass(frozen=True)
class Reallocation:
    """Decision of the dead-band planner.

    Attributes
    ----------
    distance : float
        Aitchison distance between the current and the target shares.
    reallocate : bool
        True if ``distance > delta``: the memory is moved.
    shares : numpy.ndarray
        Shares to use from now on: the target if moving, otherwise the current.
    """

    distance: float
    reallocate: bool
    shares: FloatArray


def plan(current: npt.ArrayLike, target: npt.ArrayLike, delta: float) -> Reallocation:
    """Reallocate only when the target moved by more than the dead band δ.

    Each reallocation costs a balloon cycle, so small drifts are ignored. The
    dead band is a distance in clr coordinates, not in MiB.

    Examples
    --------
    >>> plan([0.5, 0.5], [0.501, 0.499], delta=0.05).reallocate
    False
    >>> plan([0.5, 0.5], [0.7, 0.3], delta=0.05).reallocate
    True
    """
    if delta < 0.0:
        msg = "delta must be non-negative"
        raise ValueError(msg)
    dist = float(aitchison_distance(current, target))
    move = dist > delta
    shares = closure(target) if move else closure(current)
    return Reallocation(distance=dist, reallocate=move, shares=shares)
