"""Simulation of the checking loop on a Poisson demand with a step in its rate.

Demand for a move arrives as a Poisson flow with rate ``λ(t)``. The decision
of a check is 1 if at least one demand arrived since the previous check (with
an optional carry-over of unresolved demand to the next check, which breaks
the model; see experiment Э8). The scheduler chooses the next interval from
the past only.

Use :func:`simulate` for one run. For Monte Carlo over many runs and detector
run lengths use :func:`trosna_regime_control.detectors.run_lengths`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scheduler import Scheduler, Step
from .trace import Trace


@dataclass(frozen=True)
class StepIntensity:
    """Piecewise constant rate: ``rate_before`` until ``t_change``, then ``rate_after``.

    Attributes
    ----------
    rate_before : float
        Rate before the change, 1/s.
    rate_after : float, optional
        Rate after the change, 1/s. None means no change.
    t_change : float
        Time of the change, seconds.

    Examples
    --------
    >>> StepIntensity(0.1, 0.3, t_change=10.0).integral(5.0, 15.0)
    2.0
    """

    rate_before: float
    rate_after: float | None = None
    t_change: float = math.inf

    def integral(self, t0: float, t1: float) -> float:
        """Integral of the rate over ``[t0, t1]``: the mean number of demands."""
        if self.rate_after is None or not math.isfinite(self.t_change):
            return self.rate_before * (t1 - t0)
        before = max(0.0, min(t1, self.t_change) - t0)
        after = max(0.0, t1 - max(t0, self.t_change))
        return self.rate_before * before + self.rate_after * after


class UniformIntervals:
    """Scheduler with intervals drawn uniformly from ``[lo, hi]``, independent of the past.

    Variable intervals are needed to identify the slope of ``ln Δt`` in the
    model check (experiment Э1): with a constant interval the slope and the
    intercept are not separated.

    Examples
    --------
    >>> sched = UniformIntervals(2.0, 4.0, seed=1)
    >>> 2.0 <= sched.start(0.0) <= 4.0
    True
    """

    def __init__(self, lo: float = 1.0, hi: float = 10.0, seed: int | None = None) -> None:
        if not 0.0 < lo <= hi:
            msg = "the bounds must satisfy 0 < lo <= hi"
            raise ValueError(msg)
        self.lo = float(lo)
        self.hi = float(hi)
        self._rng = np.random.default_rng(seed)
        self._next = self.lo

    def _draw(self) -> float:
        return float(self._rng.uniform(self.lo, self.hi))

    def start(self, t0: float) -> float:
        """Draw and return the first interval."""
        del t0
        self._next = self._draw()
        return self._next

    def update(self, t: float, decision: bool) -> Step:
        """Draw the next interval; the decision is ignored."""
        del decision
        self._next = self._draw()
        return Step(
            t_next=float(t) + self._next,
            next_dt=self._next,
            nonstationary=False,
            alarm=None,
            rate=math.nan,
        )


def simulate(
    scheduler: Scheduler,
    intensity: StepIntensity,
    horizon_s: float,
    seed: int | None = None,
    carryover: float = 0.0,
) -> Trace:
    """Run the checking loop on a Poisson demand for ``horizon_s`` seconds.

    Parameters
    ----------
    scheduler : Scheduler
        Chooses the intervals; it sees only the past decisions.
    intensity : StepIntensity
        Demand rate.
    horizon_s : float
        Length of the run, seconds. Checks go on while they fit in it.
    seed : int, optional
        Seed of the random generator.
    carryover : float
        Probability that a demand not served at a check is carried to the
        next check (0 under the model).

    Returns
    -------
    Trace
        Check times, intervals and decisions of the run.

    Examples
    --------
    >>> from trosna_regime_control.scheduler import FixedInterval
    >>> tr = simulate(FixedInterval(2.0), StepIntensity(0.05), 100.0, seed=1)
    >>> len(tr)
    50
    """
    if not 0.0 <= carryover < 1.0:
        msg = "carryover must be in [0, 1)"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    dt = scheduler.start(0.0)
    t = 0.0
    pending = False
    times: list[float] = []
    dts: list[float] = []
    bits: list[int] = []
    while t + dt <= horizon_s:
        t_next = t + dt
        needs = int(rng.poisson(intensity.integral(t, t_next)))
        move = needs > 0 or pending
        if carryover > 0.0:
            pending = move and bool(rng.random() < carryover)
        step = scheduler.update(t_next, move)
        times.append(t_next)
        dts.append(dt)
        bits.append(int(move))
        t = t_next
        dt = step.next_dt
    if not times:
        msg = "the horizon is shorter than the first interval"
        raise ValueError(msg)
    return Trace(
        t=np.asarray(times, dtype=np.float64),
        dt=np.asarray(dts, dtype=np.float64),
        decision=np.asarray(bits, dtype=np.int64),
    )
