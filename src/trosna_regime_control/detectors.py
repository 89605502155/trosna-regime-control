"""Martingale detectors of a change in the decision intensity (idea 4).

Model: ``P(b_k = 1 | past) = 1 − e^{−λ Δt_k}``, the decision bits arrive at
irregular, policy-dependent intervals. Hypotheses: H₀: ``λ = λ₀``; H₁:
``λ = κ λ₀``. The likelihood-ratio factor of check k is

    f_k = (1 − e^{−κλ₀Δt_k}) / (1 − e^{−λ₀Δt_k})   if b_k = 1
    f_k = e^{−(κ−1)λ₀Δt_k}                          if b_k = 0

Under H₀, ``E[f_k | past] = 1``, so ``Λ_n = f_1 ⋯ f_n`` is a nonnegative
martingale even when the intervals are chosen adaptively from the past. Then:

* Ville's inequality: ``P₀(∃n: Λ_n ≥ 1/α) ≤ α`` (a one-shot test);
* Shiryaev–Roberts ``R_k = (1 + R_{k−1}) f_k``: the mean number of checks to a
  false alarm is at least the threshold A (Pollak, 1985);
* time-weighted Shiryaev–Roberts ``R_k = (Δt_k + R_{k−1}) f_k``: the mean time
  to a false alarm is at least A seconds;
* CUSUM ``S_k = max(0, S_{k−1} + ln f_k)``, alarm at ``h``; its mean run length
  is at least ``e^h``, so ``h = ln A`` is a conservative threshold.

Guarantees hold under the Poisson model. The model is checked by
:mod:`trosna_regime_control.regression`, and the consequences of its violation are
measured in the experiment script.

Literature: Shiryaev (1976); Pollak (1985); Page (1954); Lorden (1971);
Ramdas et al. (2023).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ._types import FloatArray

# --- Model and factors ------------------------------------------------------


def bernoulli_prob(rate: npt.ArrayLike, dt: npt.ArrayLike) -> FloatArray:
    """``P(b = 1) = 1 − e^{−λΔt}``.

    Examples
    --------
    >>> round(float(bernoulli_prob(0.05, 10.0)), 4)
    0.3935
    """
    lam = np.asarray(rate, dtype=np.float64)
    d = np.asarray(dt, dtype=np.float64)
    return np.asarray(-np.expm1(-lam * d), dtype=np.float64)


def lr_factor(decision: npt.ArrayLike, dt: npt.ArrayLike, lam0: float, kappa: float) -> FloatArray:
    """Likelihood-ratio factor ``f_k`` of H₁: ``λ = κλ₀`` against H₀: ``λ = λ₀``.

    Vectorised over ``decision`` and ``dt``. Requires ``lam0 > 0``.

    Examples
    --------
    >>> round(float(lr_factor(1, 1.0, 0.05, 2.0)), 4)
    1.9512
    """
    if lam0 <= 0.0 or kappa <= 0.0:
        msg = "lam0 and kappa must be positive"
        raise ValueError(msg)
    d = np.asarray(dt, dtype=np.float64)
    b = np.asarray(decision, dtype=np.float64)
    p0 = bernoulli_prob(lam0, d)
    p1 = bernoulli_prob(kappa * lam0, d)
    f_one = p1 / p0
    f_zero = np.exp(-(kappa - 1.0) * lam0 * d)
    return np.asarray(np.where(b > 0.0, f_one, f_zero), dtype=np.float64)


def expected_factor(rate: float, dt: npt.ArrayLike, lam0: float, kappa: float) -> FloatArray:
    """Return the expected factor ``E_λ[f]`` for a true rate ``λ``.

    Equals 1 at ``λ = λ₀``. For ``κ > 1`` it is at most 1 for every ``λ ≤ λ₀``,
    so the guarantee covers the whole null ``λ ≤ λ₀``.
    """
    p = bernoulli_prob(rate, dt)
    return np.asarray(
        p * lr_factor(1.0, dt, lam0, kappa) + (1.0 - p) * lr_factor(0.0, dt, lam0, kappa),
        dtype=np.float64,
    )


def kl_per_check(dt: float, lam0: float, kappa: float) -> float:
    """Information of one check after the change: ``KL(Bern(p₁) ‖ Bern(p₀))``, nats."""
    p0 = -math.expm1(-lam0 * dt)
    p1 = -math.expm1(-kappa * lam0 * dt)
    return p1 * math.log(p1 / p0) + (1.0 - p1) * math.log((1.0 - p1) / (1.0 - p0))


def kl_rate_poisson(lam0: float, kappa: float) -> float:
    """Information per second of a continuously observed Poisson flow: ``λ₀(κ ln κ − κ + 1)``.

    This is an upper bound for bit observations.

    Examples
    --------
    >>> round(kl_rate_poisson(0.05, 2.0), 5)
    0.01931
    """
    return lam0 * (kappa * math.log(kappa) - kappa + 1.0)


def lorden_delay_bound(arl: float, info: float) -> float:
    """Asymptotic lower bound of the worst detection delay, ``ln(ARL) / I`` (Lorden, 1971).

    ``arl`` and ``info`` must use consistent units (checks or seconds). The
    bound is asymptotic: at moderate ARL it is typically too high.

    Examples
    --------
    >>> round(lorden_delay_bound(3600.0, kl_rate_poisson(0.05, 2.0)))
    424
    """
    return math.log(arl) / info


def cusum_threshold_for_arl(arl: float) -> float:
    """CUSUM threshold ``h = ln(ARL)``: its mean run length is then at least ``e^h = ARL``."""
    if arl <= 1.0:
        msg = "ARL must exceed 1"
        raise ValueError(msg)
    return math.log(arl)


# --- Paths for arrays (runs × steps) ----------------------------------------


def log_lr_path(f: npt.ArrayLike) -> FloatArray:
    """``ln Λ_k = Σ_{i≤k} ln f_i`` along the last axis (Ville's test)."""
    return np.asarray(np.cumsum(np.log(np.asarray(f, dtype=np.float64)), axis=-1), dtype=np.float64)


def cusum_path(f: npt.ArrayLike) -> FloatArray:
    """CUSUM path ``S_k = max(0, S_{k−1} + ln f_k)`` along the last axis."""
    w = log_lr_path(f)
    running_min = np.minimum.accumulate(w, axis=-1)
    return np.asarray(w - np.minimum(running_min, 0.0), dtype=np.float64)


def sr_path(f: npt.ArrayLike) -> FloatArray:
    """Shiryaev–Roberts path ``R_k = (1 + R_{k−1}) f_k``, ``R_0 = 0``."""
    arr = np.asarray(f, dtype=np.float64)
    out = np.empty_like(arr)
    r = np.zeros(arr.shape[:-1])
    for k in range(arr.shape[-1]):
        r = (1.0 + r) * arr[..., k]
        out[..., k] = r
    return out


def sr_time_path(f: npt.ArrayLike, dt: npt.ArrayLike) -> FloatArray:
    """Time-weighted Shiryaev–Roberts path ``R_k = (Δt_k + R_{k−1}) f_k``."""
    arr = np.asarray(f, dtype=np.float64)
    step = np.broadcast_to(np.asarray(dt, dtype=np.float64), arr.shape)
    out = np.empty_like(arr)
    r = np.zeros(arr.shape[:-1])
    for k in range(arr.shape[-1]):
        r = (step[..., k] + r) * arr[..., k]
        out[..., k] = r
    return out


def first_passage_index(path: npt.ArrayLike, threshold: float) -> npt.NDArray[np.int64]:
    """Index of the first step where ``path ≥ threshold`` along the last axis, −1 if none."""
    hit = np.asarray(path, dtype=np.float64) >= threshold
    idx = np.argmax(hit, axis=-1)
    return np.asarray(np.where(hit.any(axis=-1), idx, -1), dtype=np.int64)


def first_passage_time(path: npt.ArrayLike, threshold: float, t: npt.ArrayLike) -> FloatArray:
    """Return the time of the first alarm, seconds; ``inf`` where there is none.

    ``t`` has the shape of ``path``.
    """
    idx = first_passage_index(path, threshold)
    times = np.asarray(t, dtype=np.float64)
    taken = np.take_along_axis(times, np.maximum(idx, 0)[..., None], axis=-1)[..., 0]
    return np.asarray(np.where(idx >= 0, taken, np.inf), dtype=np.float64)


def run_lengths(
    kind: str,
    threshold: float,
    lam_true: float,
    lam0: float,
    kappa: float,
    dt: float = 1.0,
    n_runs: int = 2000,
    max_steps: int = 1_000_000,
    seed: int | None = None,
) -> FloatArray:
    """Count checks until the first alarm, for constant ``Δt`` and rate, without traces.

    ``kind="sr"`` uses ``R = (1 + R) f``; ``kind="cusum"`` uses ``S = max(0, S + ln f)``.
    ``lam_true = lam0`` gives the ARL; ``lam_true = κ λ₀`` gives the delay when
    the change happens at the start. Runs that do not alarm within ``max_steps``
    are ``inf``. This is the Monte Carlo core; its cost is proportional to the
    mean run length times ``n_runs``.

    Examples
    --------
    >>> rl = run_lengths("sr", 50.0, 0.05, 0.05, 2.0, n_runs=200, seed=1)
    >>> bool(np.all(np.isfinite(rl)))
    True
    """
    if kind not in ("sr", "cusum"):
        msg = "kind must be 'sr' or 'cusum'"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    p_true = -math.expm1(-lam_true * dt)
    p0 = -math.expm1(-lam0 * dt)
    p1 = -math.expm1(-kappa * lam0 * dt)
    f_one, f_zero = p1 / p0, math.exp(-(kappa - 1.0) * lam0 * dt)
    l_one, l_zero = math.log(f_one), math.log(f_zero)
    stat = np.zeros(n_runs)
    idx = np.arange(n_runs)
    out = np.full(n_runs, np.inf)
    for k in range(1, max_steps + 1):
        bit = rng.random(stat.size) < p_true
        if kind == "sr":
            stat = (1.0 + stat) * np.where(bit, f_one, f_zero)
        else:
            stat = np.maximum(0.0, stat + np.where(bit, l_one, l_zero))
        hit = stat >= threshold
        if hit.any():
            out[idx[hit]] = k
            stat, idx = stat[~hit], idx[~hit]
            if stat.size == 0:
                break
    return out


# --- Online detectors -------------------------------------------------------


class ShiryaevRoberts:
    """Shiryaev–Roberts detector: alarm at ``R ≥ threshold``, then the statistic resets.

    Parameters
    ----------
    lam0 : float
        Baseline rate λ₀, 1/s.
    kappa : float
        Alternative multiplier κ (κ > 1 for a rise, κ < 1 for a fall).
    threshold : float
        Threshold A. With ``time_weighted=True`` it is in seconds, and the mean
        time to a false alarm is at least A.
    time_weighted : bool
        Use ``R = (Δt + R) f`` instead of ``R = (1 + R) f``.

    Examples
    --------
    >>> det = ShiryaevRoberts.for_arl(0.05, 2.0, arl=3600.0)
    >>> det.threshold
    3600.0
    """

    def __init__(
        self,
        lam0: float,
        kappa: float,
        threshold: float,
        time_weighted: bool = True,
        stat: float = 0.0,
    ) -> None:
        if lam0 <= 0.0 or kappa <= 0.0 or threshold <= 0.0:
            msg = "lam0, kappa and threshold must be positive"
            raise ValueError(msg)
        self.lam0 = float(lam0)
        self.kappa = float(kappa)
        self.threshold = float(threshold)
        self.time_weighted = bool(time_weighted)
        self.stat = float(stat)

    @classmethod
    def for_arl(
        cls, lam0: float, kappa: float, arl: float, time_weighted: bool = True
    ) -> ShiryaevRoberts:
        """Detector with threshold equal to the required ARL (in seconds if time-weighted)."""
        return cls(lam0, kappa, arl, time_weighted=time_weighted)

    def record_check(self, decision: float, dt: float) -> bool:
        """Account for one check. Returns True on an alarm, and then resets."""
        f = float(lr_factor(decision, dt, self.lam0, self.kappa))
        weight = dt if self.time_weighted else 1.0
        self.stat = (weight + self.stat) * f
        if self.stat >= self.threshold:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Set the statistic to zero."""
        self.stat = 0.0

    def to_dict(self) -> dict[str, Any]:
        """State as a JSON-serialisable dictionary."""
        return {
            "kind": "sr",
            "lam0": self.lam0,
            "kappa": self.kappa,
            "threshold": self.threshold,
            "time_weighted": self.time_weighted,
            "stat": self.stat,
        }


class Cusum:
    """CUSUM detector: ``S = max(0, S + ln f)``, alarm at ``S ≥ h``, then reset.

    Examples
    --------
    >>> det = Cusum(0.05, 2.0, h=cusum_threshold_for_arl(3600.0))
    >>> round(det.h, 3)
    8.189
    """

    def __init__(self, lam0: float, kappa: float, h: float, stat: float = 0.0) -> None:
        if lam0 <= 0.0 or kappa <= 0.0 or h <= 0.0:
            msg = "lam0, kappa and h must be positive"
            raise ValueError(msg)
        self.lam0 = float(lam0)
        self.kappa = float(kappa)
        self.h = float(h)
        self.stat = float(stat)

    def record_check(self, decision: float, dt: float) -> bool:
        """Account for one check. Returns True on an alarm, and then resets."""
        log_f = float(np.log(lr_factor(decision, dt, self.lam0, self.kappa)))
        self.stat = max(0.0, self.stat + log_f)
        if self.stat >= self.h:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Set the statistic to zero."""
        self.stat = 0.0

    def to_dict(self) -> dict[str, Any]:
        """State as a JSON-serialisable dictionary."""
        return {
            "kind": "cusum",
            "lam0": self.lam0,
            "kappa": self.kappa,
            "h": self.h,
            "stat": self.stat,
        }


class TwoSided:
    """Two Shiryaev–Roberts detectors: a rise (κ_up > 1) and a fall (κ_down < 1).

    The rise detector has the null ``λ ≤ λ₀⁺`` and the fall detector the null
    ``λ ≥ λ₀⁻``, so each may use its own baseline (an upper and a lower
    confidence bound). :meth:`record_check` returns ``+1`` on an alarm about a rise,
    ``−1`` about a fall, and ``0`` otherwise. After any alarm both statistics
    restart.

    Examples
    --------
    >>> two = TwoSided(0.05, threshold=3600.0)
    >>> two.record_check(0.0, 1.0)
    0
    """

    def __init__(
        self,
        lam0_up: float,
        threshold: float,
        kappa_up: float = 2.0,
        kappa_down: float = 0.5,
        time_weighted: bool = True,
        lam0_down: float | None = None,
    ) -> None:
        down0 = lam0_up if lam0_down is None else lam0_down
        self.up = ShiryaevRoberts(lam0_up, kappa_up, threshold, time_weighted)
        self.down = ShiryaevRoberts(down0, kappa_down, threshold, time_weighted)

    @property
    def lam0_up(self) -> float:
        """Baseline of the rise detector, 1/s."""
        return self.up.lam0

    @property
    def lam0_down(self) -> float:
        """Baseline of the fall detector, 1/s."""
        return self.down.lam0

    def record_check(self, decision: float, dt: float) -> int:
        """Account for one check; returns +1 (rise), −1 (fall) or 0."""
        up = self.up.record_check(decision, dt)
        down = self.down.record_check(decision, dt)
        if up or down:
            self.up.reset()
            self.down.reset()
        return 1 if up else (-1 if down else 0)
