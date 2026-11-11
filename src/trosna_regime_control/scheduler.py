"""Common interface of the schedulers: when to check next, and what the regime is.

A scheduler receives the decision of each check and returns the interval to
the next one. Three schedulers ship with the library:

* :class:`FixedInterval` — constant interval (baseline);
* :class:`~trosna_regime_control.heuristic.LegacyHeuristic` — the 5-bit heuristic of the
  current system, for comparison and migration;
* :class:`~trosna_regime_control.controller.RegimeController` — the proposed method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .trace import Trace


@dataclass(frozen=True)
class Step:
    """Result of one check.

    Attributes
    ----------
    t_next : float
        Planned time of the next check, seconds.
    next_dt : float
        Interval to the next check, seconds.
    nonstationary : bool
        Regime after this check.
    alarm : str or None
        ``"rise"`` or ``"fall"`` if a detector raised an alarm on this check.
    rate : float
        Estimate of the decision intensity used for the interval, 1/s
        (NaN if the scheduler has none).
    """

    t_next: float
    next_dt: float
    nonstationary: bool
    alarm: str | None
    rate: float


class Scheduler(Protocol):
    """Anything that starts at a time and returns a :class:`Step` on every check."""

    def start(self, t0: float) -> float:
        """Begin observation at ``t0``; return the first interval, seconds."""
        ...

    def update(self, t: float, decision: bool) -> Step:
        """Account for the check made at time ``t`` with decision ``decision``."""
        ...


@dataclass
class FixedInterval:
    """Constant interval. Used as the baseline in comparisons.

    Examples
    --------
    >>> fixed = FixedInterval(4.0)
    >>> fixed.start(0.0)
    4.0
    >>> fixed.update(4.0, True).next_dt
    4.0
    """

    dt: float

    def __post_init__(self) -> None:
        """Check that the interval is positive."""
        if self.dt <= 0.0:
            msg = "the interval must be positive"
            raise ValueError(msg)

    def start(self, t0: float) -> float:
        """Return the constant interval."""
        del t0
        return self.dt

    def update(self, t: float, decision: bool) -> Step:
        """Return the constant interval as the next step."""
        del decision
        return Step(
            t_next=t + self.dt,
            next_dt=self.dt,
            nonstationary=False,
            alarm=None,
            rate=math.nan,
        )


def replay(scheduler: Scheduler, trace: Trace) -> list[Step]:
    """Feed the recorded decisions of a trace to a scheduler, in order.

    The scheduler sees the recorded check times, so its steps show what it
    would have chosen on the same data. The first call sets the start to the
    time of the first check minus its recorded interval.

    Examples
    --------
    >>> from trosna_regime_control.trace import Trace
    >>> import numpy as np
    >>> tr = Trace(t=np.array([10.0, 20.0]), dt=np.array([10.0, 10.0]), decision=np.array([0, 1]))
    >>> [s.next_dt for s in replay(FixedInterval(5.0), tr)]
    [5.0, 5.0]
    """
    scheduler.start(float(trace.t[0] - trace.dt[0]))
    return [
        scheduler.update(float(t), bool(b)) for t, b in zip(trace.t, trace.decision, strict=True)
    ]
