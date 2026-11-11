"""Flicker of the current heuristic: exact Markov-chain analysis (idea 3).

With a constant intensity ``λ`` the 5-bit history of the current heuristic is
a Markov chain on ``2^n`` states. From a state with sum ``S`` the system waits
``Δt = table[S]`` and writes a bit ``b = 1`` with probability ``1 − e^{−λΔt}``.
From the stationary distribution the frequency of regime switches per hour
follows exactly. For the document's conditional table and threshold 3 this
gives 57 switches per hour at ``λ = 0.05 1/s`` and 137 at ``λ = 0.1 1/s``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._types import BoolArray, FloatArray, IntArray
from .heuristic import HeuristicConfig, default_config, window_sums


@dataclass(frozen=True)
class MarkovChain:
    """Transition matrix and stationary law of the heuristic at a fixed rate.

    Attributes
    ----------
    transition : numpy.ndarray
        Matrix ``P[h, h']`` on the ``2^n`` history states.
    stationary : numpy.ndarray
        Stationary distribution ``π``.
    sums : numpy.ndarray
        Sum ``S`` of each state.
    intervals : numpy.ndarray
        Interval waited from each state, seconds.
    p_one : numpy.ndarray
        Probability of a move during the wait from each state.
    mode : numpy.ndarray
        Non-stationary regime of each state (boolean).
    """

    transition: FloatArray
    stationary: FloatArray
    sums: IntArray
    intervals: FloatArray
    p_one: FloatArray
    mode: BoolArray


@dataclass(frozen=True)
class FlickerStats:
    """Long-run characteristics of the heuristic at a constant rate.

    Attributes
    ----------
    rate : float
        Intensity λ, 1/s.
    switches_per_hour : float
        Regime changes per hour.
    share_nonstationary : float
        Share of checks in the non-stationary regime.
    mean_interval_s : float
        Mean interval, seconds.
    checks_per_hour : float
        Checks per hour.
    share_moves : float
        Share of checks that move memory.
    """

    rate: float
    switches_per_hour: float
    share_nonstationary: float
    mean_interval_s: float
    checks_per_hour: float
    share_moves: float


def markov_chain(rate: float, config: HeuristicConfig | None = None) -> MarkovChain:
    """Build the chain of the heuristic for the intensity ``rate``, 1/s."""
    cfg = config if config is not None else default_config()
    n_states = 2**cfg.n_bits
    mask = n_states - 1
    states = np.arange(n_states)
    sums = np.array([(h).bit_count() for h in states], dtype=np.int64)
    intervals = np.array([cfg.interval(int(s)) for s in sums], dtype=np.float64)
    p1 = -np.expm1(-rate * intervals)
    trans = np.zeros((n_states, n_states))
    trans[states, ((states << 1) | 1) & mask] += p1
    trans[states, (states << 1) & mask] += 1.0 - p1
    # Stationary law: πP = π with Σπ = 1, solved as an overdetermined linear system.
    system = np.vstack([trans.T - np.eye(n_states), np.ones(n_states)])
    rhs = np.zeros(n_states + 1)
    rhs[-1] = 1.0
    pi = np.linalg.lstsq(system, rhs, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()
    mode = np.array([cfg.mode(int(s)) for s in sums], dtype=bool)
    return MarkovChain(
        transition=trans,
        stationary=np.asarray(pi, dtype=np.float64),
        sums=sums,
        intervals=intervals,
        p_one=np.asarray(p1, dtype=np.float64),
        mode=mode,
    )


def flicker_stats(rate: float, config: HeuristicConfig | None = None) -> FlickerStats:
    """Exact flicker of the heuristic at a constant rate.

    Examples
    --------
    >>> round(flicker_stats(0.05).switches_per_hour, 1)
    57.4
    """
    mc = markov_chain(rate, config)
    pi = mc.stationary
    changes = mc.mode[:, None] != mc.mode[None, :]
    switches_per_step = float(pi @ (mc.transition * changes).sum(axis=1))
    mean_dt = float(pi @ mc.intervals)
    return FlickerStats(
        rate=float(rate),
        switches_per_hour=switches_per_step / mean_dt * 3600.0,
        share_nonstationary=float(pi @ mc.mode),
        mean_interval_s=mean_dt,
        checks_per_hour=3600.0 / mean_dt,
        share_moves=float(pi @ mc.p_one),
    )


def switches_per_hour(
    decisions: Sequence[int] | IntArray,
    dt: Sequence[float] | FloatArray,
    config: HeuristicConfig | None = None,
    burn_in: int = 0,
) -> float:
    """Regime switches per hour of the heuristic on a recorded decision sequence.

    The regime after check ``k`` is computed from the window of the decisions
    ``1…k``; the first ``burn_in`` checks are skipped.

    Examples
    --------
    >>> switches_per_hour([0, 0, 0, 1, 1, 1, 1, 0, 0], [1.0] * 9, burn_in=0) > 0
    True
    """
    cfg = config if config is not None else default_config()
    bits = np.asarray(decisions, dtype=np.int64)
    step = np.asarray(dt, dtype=np.float64)
    sums = window_sums(bits.tolist(), cfg.n_bits)
    mode = np.array([cfg.mode(int(s)) for s in sums], dtype=bool)
    if mode.shape[0] - burn_in < 2:
        msg = "too few checks after the burn-in"
        raise ValueError(msg)
    flips = int(np.count_nonzero(mode[burn_in + 1 :] != mode[burn_in:-1]))
    duration = float(step[burn_in + 1 :].sum())
    if duration <= 0.0:
        msg = "zero observation time"
        raise ValueError(msg)
    return flips / duration * 3600.0
