"""Validation of one check before it enters the controller.

:func:`validate_decision` is the single public check of an incoming check. It is
used by :meth:`~trosna_regime_control.controller.RegimeController.record_decision`,
so a bad check is refused before any state changes.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = ["validate_decision"]


def _real_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"t must be a real number, got {type(value).__name__} {value!r}"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"t must be finite, got {number!r}"
        raise ValueError(msg)
    if number < 0.0:
        msg = f"t must be non-negative, got {number!r}"
        raise ValueError(msg)
    return number


def _decision(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        if value in (0, 1):
            return bool(value)
        msg = f"decision must be 0 or 1, got {value!r}"
        raise ValueError(msg)
    msg = f"decision must be a bool or the integer 0 or 1, got {type(value).__name__} {value!r}"
    raise TypeError(msg)


def validate_decision(
    t: object,
    decision: object,
    last_t: float | None = None,
) -> tuple[float, bool]:
    """Check one check ``(t, decision)`` of the controller and return it as ``(float, bool)``.

    Parameters
    ----------
    t : real number
        Time of the check, seconds. Finite and non-negative. Must be later than
        ``last_t`` when that is given.
    decision : bool or int
        True (or 1) if memory was moved at this check, False (or 0) if not.
        A numpy boolean is accepted. Floats are refused: a fractional decision
        is not a Bernoulli outcome.
    last_t : float, optional
        Time of the previous check; ``None`` for the first one after ``start``.

    Returns
    -------
    tuple of (float, bool)
        ``(t, decision)`` converted to Python types.

    Raises
    ------
    TypeError
        If ``t`` is not a real number, or ``decision`` is not a bool or an integer
        (a bool, a string, ``None``, a float, and similar).
    ValueError
        If ``t`` is NaN, infinite or negative, if ``t`` does not increase, or if
        an integer decision is not 0 or 1.

    Examples
    --------
    >>> validate_decision(10, 1)
    (10.0, True)
    >>> validate_decision(5.0, False, last_t=10.0)
    Traceback (most recent call last):
    ...
    ValueError: time must increase: 5.0 after 10.0
    >>> validate_decision(1.0, 0.5)
    Traceback (most recent call last):
    ...
    TypeError: decision must be a bool or the integer 0 or 1, got float 0.5
    """
    time = _real_time(t)
    bit = _decision(decision)
    if last_t is not None and time <= last_t:
        msg = f"time must increase: {time!r} after {last_t!r}"
        raise ValueError(msg)
    return time, bit
