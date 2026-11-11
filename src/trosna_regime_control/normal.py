"""Standard normal quantiles without SciPy: the confidence bound of the baselines.

The baselines of the detectors are upper and lower confidence bounds of the
naive rate (:meth:`~trosna_regime_control.intensity.ForgettingRate.upper_naive`).
Their width is set by a one-sided confidence level ``c``, converted here to the
number of standard deviations ``z = Φ⁻¹(c)``.

The quantile uses Wichura's algorithm AS 241 (PPND16, Wichura 1988), accurate to
about 1e-16 relative error over the whole open interval ``(0, 1)``.

Examples
--------
>>> round(z_for_confidence(0.95), 6)
1.644854
"""

from __future__ import annotations

import math

__all__ = ["baseline_z", "check_z", "norm_ppf", "z_for_confidence"]


def _tail_quantile(r: float) -> float:
    """Quantile for ``r = √(−ln min(p, 1 − p))`` (AS 241, the two outer branches)."""
    if r <= 5.0:
        t = r - 1.6
        num = (
            (
                (
                    (
                        (
                            (t * 7.7454501427834140764e-4 + 0.0227238449892691845833) * t
                            + 0.24178072517745061177
                        )
                        * t
                        + 1.27045825245236838258
                    )
                    * t
                    + 3.64784832476320460504
                )
                * t
                + 5.7694972214606914055
            )
            * t
            + 4.6303378461565452959
        ) * t + 1.42343711074968357734
        den = (
            (
                (
                    (
                        (
                            (t * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * t
                            + 0.0151986665636164571966
                        )
                        * t
                        + 0.14810397642748007459
                    )
                    * t
                    + 0.68976733498510000455
                )
                * t
                + 1.6763848301838038494
            )
            * t
            + 2.05319162663775882187
        ) * t + 1.0
        return num / den
    t = r - 5.0
    num = (
        (
            (
                (
                    (
                        (t * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * t
                        + 0.0012426609473880784386
                    )
                    * t
                    + 0.026532189526576123093
                )
                * t
                + 0.29656057182850489123
            )
            * t
            + 1.7848265399172913358
        )
        * t
        + 5.4637849111641143699
    ) * t + 6.6579046435011037772
    den = (
        (
            (
                (
                    (
                        (t * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * t
                        + 1.8463183175100546818e-5
                    )
                    * t
                    + 7.868691311456132591e-4
                )
                * t
                + 0.0148753612908506148525
            )
            * t
            + 0.13692988092273580531
        )
        * t
        + 0.59983220655588793769
    ) * t + 1.0
    return num / den


def norm_ppf(p: float) -> float:
    """Quantile function ``Φ⁻¹(p)`` of the standard normal distribution.

    Parameters
    ----------
    p : float
        Probability, ``0 < p < 1``.

    Raises
    ------
    TypeError
        If ``p`` is not a real number (a bool, a string, ``None``, and similar).
    ValueError
        If ``p`` is not strictly between 0 and 1.

    Examples
    --------
    >>> round(norm_ppf(0.975), 6)
    1.959964
    >>> norm_ppf(0.5)
    0.0
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        msg = f"p must be a real number, got {type(p).__name__} {p!r}"
        raise TypeError(msg)
    if not 0.0 < p < 1.0:
        msg = f"p must lie strictly between 0 and 1, got {p!r}"
        raise ValueError(msg)
    q = float(p) - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (
            (
                (
                    (
                        (
                            (r * 2509.0809287301226727 + 33430.575583588128105) * r
                            + 67265.770927008700853
                        )
                        * r
                        + 45921.953931549871457
                    )
                    * r
                    + 13731.693765509461125
                )
                * r
                + 1971.5909503065514427
            )
            * r
            + 133.14166789178437745
        ) * r + 3.387132872796366608
        den = (
            (
                (
                    (
                        (
                            (r * 5226.495278852545925 + 28729.085735721942674) * r
                            + 39307.89580009271061
                        )
                        * r
                        + 21213.794301586595867
                    )
                    * r
                    + 5394.1960214247511077
                )
                * r
                + 687.1870074920579083
            )
            * r
            + 42.313330701600911252
        ) * r + 1.0
        return q * num / den
    tail = min(float(p), 1.0 - float(p))
    value = _tail_quantile(math.sqrt(-math.log(tail)))
    return -value if q < 0.0 else value


def check_z(z: object) -> float:
    """Return ``z`` as a float after checking that it is a finite non-negative number.

    Raises
    ------
    TypeError
        If ``z`` is not a real number.
    ValueError
        If ``z`` is NaN, infinite or negative.

    Examples
    --------
    >>> check_z(2)
    2.0
    >>> check_z(-1.0)
    Traceback (most recent call last):
    ...
    ValueError: z must be finite and non-negative, got -1.0
    """
    if isinstance(z, bool) or not isinstance(z, (int, float)):
        msg = f"z must be a real number, got {type(z).__name__} {z!r}"
        raise TypeError(msg)
    number = float(z)
    if not math.isfinite(number) or number < 0.0:
        msg = f"z must be finite and non-negative, got {number!r}"
        raise ValueError(msg)
    return number


def baseline_z(confidence: float = 0.95, z: float | None = None) -> float:
    """Effective number of standard deviations of the baseline bounds.

    ``z`` is used directly if it is given; otherwise it is ``z_for_confidence``
    of ``confidence``. Both arguments are checked.

    Examples
    --------
    >>> round(baseline_z(0.95), 4)
    1.6449
    >>> baseline_z(z=1.0)
    1.0
    """
    if z is not None:
        return check_z(z)
    return z_for_confidence(confidence)


def z_for_confidence(confidence: float) -> float:
    """Return the standard deviations of a one-sided bound at confidence ``c``.

    ``c = 0.95`` gives ``z = 1.6449`` (the constant 1.645 of the earlier draft).
    The confidence must lie in ``[0.5, 1)``: below 0.5 the bound would lie under
    the mean, and 1 would need an infinite ``z``.

    Parameters
    ----------
    confidence : float
        One-sided confidence level, ``0.5 ≤ c < 1``.

    Raises
    ------
    TypeError
        If the level is not a real number.
    ValueError
        If the level is outside ``[0.5, 1)``.

    Examples
    --------
    >>> round(z_for_confidence(0.99), 4)
    2.3263
    >>> z_for_confidence(0.5)
    0.0
    """
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        msg = f"confidence must be a real number, got {type(confidence).__name__} {confidence!r}"
        raise TypeError(msg)
    if not 0.5 <= float(confidence) < 1.0:
        msg = f"confidence must lie in [0.5, 1), got {confidence!r}"
        raise ValueError(msg)
    return norm_ppf(float(confidence))
