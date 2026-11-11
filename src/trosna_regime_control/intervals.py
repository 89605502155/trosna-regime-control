"""Monitoring interval from scale-invariant rules (idea 2).

The rule must give the same result for any memory size of the host and any
unit of time. Therefore it is built from dimensionless quantities and a cost
model, not from a hand-made table.

* Rule A: ``Δt = g(p*) / λ̂``, with ``g(p) = −ln(1 − p)`` (exact, default) or
  ``g(p) = p`` (linear, first-order approximation). The exact form makes the
  probability of a move in ``Δt`` equal to p* under a Poisson model.
* Rule B: ``Δt = √(2 C₁ / (λ̂ C₂))``. Minimises the cost rate
  ``J(Δt) = C₁/Δt + λ C₂ Δt / 2`` (C₁ cost of a check, C₂ cost of a second of
  reaction delay). Only the ratio ``C₁/C₂`` matters.
* Rule C: ``Δt = δ / v``. The distribution cannot drift farther than δ
  (in the Aitchison distance) before the next check.

Every rule is clipped to the bounds of :class:`IntervalBounds`. The default
bounds ``[1 s, 10 s]`` are those of the current heuristic.

References: Barlow, Hunter and Proschan (1963) for the exact cost optimum;
Anta and Tabuada (2010) for self-triggered sampling.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import FloatArray
from .intensity import check_estimator


@dataclass(frozen=True)
class IntervalBounds:
    """Lower and upper bounds of the monitoring interval, seconds.

    Examples
    --------
    >>> float(IntervalBounds().clip(0.2))
    1.0
    """

    lo: float = 1.0
    hi: float = 10.0

    def __post_init__(self) -> None:
        """Check the bounds."""
        if not (0.0 < self.lo <= self.hi):
            msg = "interval bounds must satisfy 0 < lo <= hi"
            raise ValueError(msg)

    def clip(self, dt: npt.ArrayLike) -> FloatArray:
        """Clip intervals to ``[lo, hi]``."""
        return np.clip(np.asarray(dt, dtype=np.float64), self.lo, self.hi)


def _ratio_rule(
    numerator: float,
    denominator: npt.ArrayLike,
    bounds: IntervalBounds | None,
    fallback: float | None,
    power: float,
) -> FloatArray:
    if bounds is None:
        bounds = IntervalBounds()
    den = np.asarray(denominator, dtype=np.float64)
    safe = np.where(den > 0.0, den, 1.0)
    raw = np.where(den > 0.0, (numerator / safe) ** power, bounds.hi)
    if fallback is None:
        fallback = bounds.hi
    raw = np.where(np.isnan(den), fallback, raw)
    return bounds.clip(raw)


RULE_A_FORMULAS: tuple[str, ...] = ("exact", "linear")


def check_rule_a(p_star: object, formula: object, scale: object) -> None:
    """Validate the parameters of rule A.

    Raises
    ------
    TypeError
        If ``p_star`` or ``scale`` is not a real number, or ``formula`` is not a string.
    ValueError
        If ``p_star`` is not in ``(0, 1)``, ``formula`` is not in
        :data:`RULE_A_FORMULAS`, or ``scale`` is not finite and positive.

    Examples
    --------
    >>> check_rule_a(0.3, "exact", 1.0)
    >>> check_rule_a(0.3, "log", 1.0)
    Traceback (most recent call last):
    ...
    ValueError: formula must be 'exact' or 'linear'
    """
    if not isinstance(p_star, numbers.Real):
        msg = "p_star must be a real number"
        raise TypeError(msg)
    if not 0.0 < float(p_star) < 1.0:
        msg = "p_star must be in (0, 1)"
        raise ValueError(msg)
    if not isinstance(formula, str):
        msg = "formula must be a string"
        raise TypeError(msg)
    if formula not in RULE_A_FORMULAS:
        msg = "formula must be 'exact' or 'linear'"
        raise ValueError(msg)
    if not isinstance(scale, numbers.Real):
        msg = "scale must be a real number"
        raise TypeError(msg)
    if not (math.isfinite(float(scale)) and float(scale) > 0.0):
        msg = "scale must be finite and positive"
        raise ValueError(msg)


def rule_a(
    rate: npt.ArrayLike,
    p_star: float,
    bounds: IntervalBounds | None = None,
    fallback: float | None = None,
    formula: str = "exact",
    scale: float = 1.0,
) -> FloatArray:
    """Rule A: ``Δt = scale · g(p*) / λ̂``, with ``g`` set by ``formula``.

    ``formula="exact"`` (default) takes ``g(p) = −ln(1 − p)``. It solves
    ``P(N(Δt) ≥ 1) = 1 − e^{−λΔt} = p*`` for ``Δt`` under a Poisson model with
    rate ``λ``. ``formula="linear"`` takes ``g(p) = p``, the first-order Taylor
    term. It is smaller than the exact value: at ``p* = 0.3`` the exact interval
    is about 19 % longer.

    With the estimate ``λ̂`` of :class:`~trosna_regime_control.intensity.ForgettingRate`,
    the share of checks that move memory is close to ``p*``. A rate of zero gives
    the upper bound; a missing rate (NaN) gives ``fallback``.

    Parameters
    ----------
    rate : array_like
        Intensity estimate λ̂, 1/s.
    p_star : float
        Target probability of at least one move in the interval, in ``(0, 1)``.
    bounds : IntervalBounds, optional
        Clipping bounds. Default ``[1 s, 10 s]``.
    fallback : float, optional
        Value used where the rate is NaN. Default: the upper bound.
    formula : {"exact", "linear"}
        Form of ``g``. Default ``"exact"``.
    scale : float
        Positive multiplier of the interval (a unit or safety factor). Default 1.

    Raises
    ------
    TypeError, ValueError
        If an argument is invalid. See :func:`check_rule_a`.

    Examples
    --------
    >>> round(float(rule_a(0.05, 0.3)), 2)
    7.13
    >>> round(float(rule_a(0.05, 0.3, formula="linear")), 2)
    6.0
    >>> float(rule_a(0.0, 0.3))
    10.0
    """
    check_rule_a(p_star, formula, scale)
    p = float(p_star)
    g = -math.log1p(-p) if formula == "exact" else p
    return _ratio_rule(scale * g, rate, bounds, fallback, power=1.0)


def rule_b(
    rate: npt.ArrayLike,
    c1_over_c2: float,
    bounds: IntervalBounds | None = None,
    fallback: float | None = None,
) -> FloatArray:
    """Rule B: ``Δt = √(2 C₁ / (λ̂ C₂))``, the square-root law of checking.

    Parameters
    ----------
    rate : array_like
        Intensity estimate λ̂, 1/s.
    c1_over_c2 : float
        Ratio of the cost of one check to the cost of one second of delay, s.

    Examples
    --------
    >>> round(float(rule_b(0.05, 2.0)), 2)
    8.94
    """
    if c1_over_c2 <= 0.0:
        msg = "C1/C2 must be positive"
        raise ValueError(msg)
    return _ratio_rule(2.0 * c1_over_c2, rate, bounds, fallback, power=0.5)


def rule_c(
    drift: npt.ArrayLike,
    delta: float,
    bounds: IntervalBounds | None = None,
    fallback: float | None = None,
) -> FloatArray:
    """Rule C: ``Δt = δ / v``, with ``v`` the smoothed drift of the shares, 1/s.

    Examples
    --------
    >>> round(float(rule_c(0.1, 0.3)), 1)
    3.0
    """
    if delta <= 0.0:
        msg = "delta must be positive"
        raise ValueError(msg)
    return _ratio_rule(delta, drift, bounds, fallback, power=1.0)


def cost_rate(dt: npt.ArrayLike, rate: float, c1: float, c2: float) -> FloatArray:
    """Cost per second ``J(Δt) = C₁/Δt + λ C₂ Δt / 2`` of the checking policy."""
    d = np.asarray(dt, dtype=np.float64)
    return np.asarray(c1 / d + rate * c2 * d / 2.0, dtype=np.float64)


def exact_interval(rate: float, c1_over_c2: float, hi: float = 1.0e6) -> float:
    """Exact optimum of the checking problem for exponential time to failure.

    Solves ``e^{λx} − 1 − λx = λ C₁/C₂`` for ``x`` by bisection (Barlow, Hunter
    and Proschan, 1963). Rule B is its first-order approximation for small
    ``λx``. Only ``rate > 0`` is accepted.

    Examples
    --------
    >>> round(exact_interval(0.05, 2.0), 2)
    8.32
    """
    if rate <= 0.0 or c1_over_c2 <= 0.0:
        msg = "rate and C1/C2 must be positive"
        raise ValueError(msg)
    target = rate * c1_over_c2

    def gap(x: float) -> float:
        return float(np.expm1(rate * x) - rate * x - target)

    lo, up = 0.0, 1.0
    while gap(up) < 0.0:
        up *= 2.0
        if up > hi:
            msg = "no solution below the search limit"
            raise ValueError(msg)
    for _ in range(200):
        mid = 0.5 * (lo + up)
        if gap(mid) < 0.0:
            lo = mid
        else:
            up = mid
    return 0.5 * (lo + up)


def steady_state_rule_a(
    rate: float,
    p_star: float,
    variant: str = "p_over_rate",
    bounds: IntervalBounds | None = None,
    n_iter: int = 1000,
) -> tuple[float, float]:
    """Steady state of rule A under a constant true rate, with the bias of ``N/T``.

    The naive estimate has the mean ``(1 − e^{−λΔt}) / Δt``. Iterating the
    rule on that mean gives the interval and the share of checks that move
    memory. ``variant="p_over_rate"`` is the linear rule A (``formula="linear"``):
    the share is exactly ``p*``. ``variant="log"`` is the exact rule A (the default):
    ``−ln(1 − p*)/λ̂``, which is right only for the true rate. Its share is about
    0.36 instead of 0.3, because the naive estimate is biased low.

    Returns
    -------
    tuple of float
        ``(Δt, share)``: the interval in seconds and the share of ones.

    Examples
    --------
    >>> dt, share = steady_state_rule_a(0.05, 0.3)
    >>> round(share, 3)
    0.3
    """
    if bounds is None:
        bounds = IntervalBounds()
    if variant == "p_over_rate":
        c = p_star
    elif variant == "log":
        c = float(-np.log1p(-p_star))
    else:
        msg = "variant must be 'p_over_rate' or 'log'"
        raise ValueError(msg)
    dt = 0.5 * (bounds.lo + bounds.hi)
    for _ in range(n_iter):
        lam_hat = float(-np.expm1(-rate * dt) / dt)
        dt = float(bounds.clip(c / lam_hat))
    return dt, float(-np.expm1(-rate * dt))


def steady_state_share(
    p_star: float,
    formula: str = "exact",
    estimator: str = "debiased",
    scale: float = 1.0,
) -> tuple[float, float]:
    """Analytic steady state of rule A for one of the four combinations, constant true rate.

    The rule gives ``Δt = scale · g(p*) / λ̂``. Let ``q = 1 − e^{−λΔt}`` be the
    share of moves. At the level of expectations the estimates are
    ``E[λ̂] = q/Δt`` for ``estimator="naive"`` and ``λ̂ = −ln(1 − q)/Δt`` for
    ``estimator="debiased"``. The fixed point gives:

    * naive: ``q = G`` and ``λΔt = −ln(1 − G)``;
    * debiased: ``−ln(1 − q) = G`` and ``λΔt = G``,

    where ``G = scale · g(p*)``. The pairs ``("exact", "debiased")`` and
    ``("linear", "naive")`` both give ``q = p*`` and ``λΔt = −ln(1 − p*)``, which is
    the target. The other two pairs miss it. The result holds while the interval
    lies inside :class:`IntervalBounds`.

    Parameters
    ----------
    p_star : float
        Target share, in ``(0, 1)``.
    formula : {"exact", "linear"}
        Form of rule A. Default ``"exact"``.
    estimator : {"naive", "debiased"}
        Rate estimator. Default ``"debiased"``.
    scale : float
        Positive multiplier of the interval. Default 1.

    Returns
    -------
    tuple of float
        ``(share, λΔt)``: the steady share of moves and the product of the rate and
        the mean interval. The mean interval is ``λΔt / λ``.

    Raises
    ------
    TypeError, ValueError
        If an argument is invalid. For the naive estimator with ``scale · g(p*) ≥ 1``
        there is no steady state, and ``ValueError`` is raised.

    Examples
    --------
    >>> share, lam_dt = steady_state_share(0.3, "exact", "naive")
    >>> round(share, 4), round(lam_dt, 4)
    (0.3567, 0.4411)
    >>> share, lam_dt = steady_state_share(0.3, "exact", "debiased")
    >>> round(share, 4), round(lam_dt, 4)
    (0.3, 0.3567)
    """
    check_rule_a(p_star, formula, scale)
    check_estimator(estimator)
    g = float(scale) * (-math.log1p(-float(p_star)) if formula == "exact" else float(p_star))
    if estimator == "naive":
        if g >= 1.0:
            msg = "no steady state: scale * g(p*) must be below 1 for the naive estimator"
            raise ValueError(msg)
        return g, -math.log1p(-g)
    return -math.expm1(-g), g
