"""Decision intensity with exponential forgetting in real time (idea 3).

Each check k brings a decision bit ``b_k`` (1: memory was moved) and the real
interval ``Δt_k`` since the previous check. The estimator keeps three numbers
and forgets them at the rate of real time::

    N_k = e^{−Δt_k/τ} N_{k−1} + b_k          weighted count of moves
    T_k = e^{−Δt_k/τ} T_{k−1} + Δt_k         weighted observation time
    K_k = e^{−Δt_k/τ} K_{k−1} + 1            weighted count of checks
    λ̂_k = N_k / T_k                          naive rate, moves per second

Exponential weights are the only choice that is consistent with a linear
one-number recursion at any intervals: the weight of a sum of ages must be the
product of the weights (Cauchy's functional equation). The naive rate is biased
low when ``λΔt`` is large, because one check records at most one move. The
debiased rate inverts ``P(b = 1) = 1 − e^{−λΔt}`` at the mean interval.

Examples
--------
>>> est = ForgettingRate(tau=600.0)
>>> for _ in range(3):
...     _ = est.update(1, 5.0)
>>> round(est.naive, 3)
0.2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


class ForgettingRate:
    """Recursive estimate of the intensity of decisions with time forgetting.

    Parameters
    ----------
    tau : float
        Memory time τ, seconds. Observations older than about τ weigh little.
    prior_rate : float, optional
        Starting estimate, 1/s, counted as ``prior_time`` seconds of data.
    prior_time : float
        Pseudo-observation time of the prior, seconds.
    prior_dt : float
        Interval of the pseudo-checks of the prior, seconds.
    """

    def __init__(
        self,
        tau: float,
        prior_rate: float | None = None,
        prior_time: float = 0.0,
        prior_dt: float = 5.0,
    ) -> None:
        if tau <= 0.0:
            msg = "tau must be positive"
            raise ValueError(msg)
        if prior_time < 0.0 or prior_dt <= 0.0:
            msg = "prior_time must be >= 0 and prior_dt > 0"
            raise ValueError(msg)
        self.tau = float(tau)
        self.n_moves = 0.0
        rate0 = 0.0 if prior_rate is None else float(prior_rate)
        self.n_moves = rate0 * prior_time
        self.exposure = float(prior_time)
        self.n_checks = prior_time / prior_dt if prior_time > 0.0 else 0.0

    def update(self, decision: float, dt: float) -> float:
        """Account for one check and return the naive rate.

        Parameters
        ----------
        decision : float
            Decision bit of the check, 0 or 1. A value between 0 and 1 is an
            expectation and is accepted for analytic tests.
        dt : float
            Real time since the previous check, seconds (positive).
        """
        if dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if not 0.0 <= decision <= 1.0:
            msg = "decision must be in [0, 1]"
            raise ValueError(msg)
        decay = math.exp(-dt / self.tau)
        self.n_moves = decay * self.n_moves + decision
        self.exposure = decay * self.exposure + dt
        self.n_checks = decay * self.n_checks + 1.0
        return self.naive

    @property
    def naive(self) -> float:
        """Naive rate ``N/T``, 1/s. NaN before any data. Biased low for large ``λΔt``."""
        if self.exposure <= 0.0:
            return math.nan
        return self.n_moves / self.exposure

    @property
    def debiased(self) -> float:
        """Debiased rate ``−ln(1 − p̂) / Δ̄``, 1/s.

        ``p̂ = N/K`` is the share of moves and ``Δ̄ = T/K`` the mean interval.
        The estimate is exact for a constant interval. The share is capped at
        ``1 − 10⁻⁹`` to keep the logarithm finite. NaN before any data.
        """
        if self.n_checks <= 0.0 or self.exposure <= 0.0:
            return math.nan
        p = min(max(self.n_moves / self.n_checks, 0.0), 1.0 - 1e-9)
        dbar = self.exposure / self.n_checks
        return -math.log1p(-p) / dbar

    def upper_naive(self, z: float = 1.645) -> float:
        """Upper confidence bound of the naive rate, 1/s: ``(N + z√N + z²/2) / T``.

        Uses the one-sided Poisson approximation with ``z`` standard deviations
        (1.645 gives 95 %). The bound is used as a conservative baseline λ₀⁺:
        the expected factor is at most 1 for every ``λ ≤ λ₀``, so a larger
        baseline keeps the false-alarm guarantee. NaN before any data.

        Examples
        --------
        >>> est = ForgettingRate(tau=600.0)
        >>> _ = est.update(1.0, 100.0)
        >>> round(est.upper_naive(1.0), 4)
        0.025
        """
        if self.exposure <= 0.0:
            return math.nan
        n = max(self.n_moves, 0.0)
        return (n + z * math.sqrt(n) + z * z / 2.0) / self.exposure

    def lower_naive(self, z: float = 1.645) -> float:
        """Lower confidence bound of the naive rate, 1/s: ``max(0, N − z√N) / T``.

        Used as the baseline of the fall detector: its null is ``λ ≥ λ₀``, so a
        smaller baseline is conservative. NaN before any data.

        Examples
        --------
        >>> est = ForgettingRate(tau=600.0)
        >>> _ = est.update(1.0, 100.0)
        >>> est.lower_naive(1.0)
        0.0
        """
        if self.exposure <= 0.0:
            return math.nan
        n = max(self.n_moves, 0.0)
        return max(0.0, n - z * math.sqrt(n)) / self.exposure

    def to_dict(self) -> dict[str, Any]:
        """State as a JSON-serialisable dictionary."""
        return {
            "tau": self.tau,
            "n_moves": self.n_moves,
            "exposure": self.exposure,
            "n_checks": self.n_checks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForgettingRate:
        """Restore the state written by :meth:`to_dict`."""
        est = cls(float(data["tau"]))
        est.n_moves = float(data["n_moves"])
        est.exposure = float(data["exposure"])
        est.n_checks = float(data["n_checks"])
        return est


def expected_naive_rate(rate: npt.ArrayLike, dt: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Mean of the naive estimate at a constant rate and interval: ``(1 − e^{−λΔt}) / Δt``.

    Examples
    --------
    >>> round(float(expected_naive_rate(0.05, 10.0)), 4)
    0.0393
    """
    lam = np.asarray(rate, dtype=np.float64)
    d = np.asarray(dt, dtype=np.float64)
    return np.asarray(-np.expm1(-lam * d) / d, dtype=np.float64)


def naive_bias(rate: float, dt: float) -> float:
    """Relative bias of the naive estimate: ``(1 − e^{−λΔt}) / (λΔt) − 1``.

    Examples
    --------
    >>> round(naive_bias(0.05, 5.0), 3)
    -0.115
    """
    x = rate * dt
    return float(-np.expm1(-x) / x - 1.0)


def relative_spread(rate: float, tau: float) -> float:
    """Approximate relative spread of the estimate for continuous observation: ``1/√(2λτ)``."""
    return float(1.0 / math.sqrt(2.0 * rate * tau))
