"""Analysis of a recorded trace: rates, model check, detector and flicker summary.

:func:`analyze` is the engine of the ``trosna-regime-control analyze`` command. It replays
the proposed controller and the current heuristic on the same recorded
decisions, so that the two regimes can be compared on real data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ._types import BoolArray
from .controller import ControllerConfig, RegimeController
from .heuristic import HeuristicConfig, default_config, window_sums
from .intensity import ForgettingRate
from .regression import CloglogFit, fit_cloglog
from .scheduler import replay
from .trace import Trace


@dataclass(frozen=True)
class TraceReport:
    """Summary of a trace. Rates are per second, switches per hour.

    Attributes
    ----------
    n_checks : int
        Number of checks.
    duration_s : float
        Time from the start to the last check.
    share_moves : float
        Share of checks with a move.
    mean_interval_s : float
        Mean real interval.
    naive_rate, debiased_rate : float
        Rate estimates at the end of the trace (memory ``tau_s``).
    cloglog : CloglogFit or None
        Model check, None if the trace is too short.
    controller_alarms : int
        Alarms of the replayed controller.
    controller_switches_per_hour : float
        Regime changes of the replayed controller.
    controller_share_nonstationary : float
        Share of checks in the non-stationary regime for the controller.
    legacy_switches_per_hour : float
        Regime changes of the current heuristic (logged, or recomputed).
    legacy_share_nonstationary : float
        Share of checks in the non-stationary regime for the current heuristic.
    """

    n_checks: int
    duration_s: float
    share_moves: float
    mean_interval_s: float
    naive_rate: float
    debiased_rate: float
    cloglog: CloglogFit | None
    controller_alarms: int
    controller_switches_per_hour: float
    controller_share_nonstationary: float
    legacy_switches_per_hour: float
    legacy_share_nonstationary: float

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        data = asdict(self)
        data["cloglog"] = None if self.cloglog is None else _fit_dict(self.cloglog)
        return data


def _fit_dict(fit: CloglogFit) -> dict[str, Any]:
    def coef(c: Any) -> dict[str, Any]:
        return {"estimate": c.estimate, "ci95": list(c.ci95)}

    return {
        "intercept_ln_lambda": coef(fit.intercept),
        "slope_ln_dt": coef(fit.slope_ln_dt),
        "coef_prev_bit": coef(fit.coef_prev_bit),
        "rate_hat": fit.rate_hat,
        "n_obs": fit.n_obs,
        "converged": fit.converged,
        "model_consistent": fit.model_consistent,
    }


def _switch_rate(mode: BoolArray, t: np.ndarray) -> tuple[float, float]:
    if mode.shape[0] < 2:
        return 0.0, 0.0
    flips = int(np.count_nonzero(mode[1:] != mode[:-1]))
    span = float(t[-1] - t[0])
    per_hour = flips / span * 3600.0 if span > 0.0 else 0.0
    return per_hour, float(mode.mean())


def analyze(
    trace: Trace,
    controller_config: ControllerConfig | None = None,
    legacy_config: HeuristicConfig | None = None,
    burn_in: int = 10,
) -> TraceReport:
    """Summarise a trace and replay both regime methods on it.

    Parameters
    ----------
    trace : Trace
        Recorded checks.
    controller_config : ControllerConfig, optional
        Parameters of the proposed controller.
    legacy_config : HeuristicConfig, optional
        Table of the current heuristic. Used only if the trace has no logged
        regime column.
    burn_in : int
        Checks skipped by the model check.
    """
    cfg = controller_config if controller_config is not None else ControllerConfig()
    legacy = legacy_config if legacy_config is not None else default_config()
    est = ForgettingRate(cfg.tau_s)
    for b, dt in zip(trace.decision, trace.dt, strict=True):
        est.update(float(b), float(dt))

    fit: CloglogFit | None
    try:
        fit = fit_cloglog(trace.decision, trace.dt, burn_in=min(burn_in, max(len(trace) - 6, 0)))
    except ValueError:
        fit = None

    ctrl = RegimeController(cfg)
    steps = replay(ctrl, trace)
    nonstat = np.array([s.nonstationary for s in steps], dtype=bool)
    alarms = sum(1 for s in steps if s.alarm is not None)
    ctrl_rate, ctrl_share = _switch_rate(nonstat, trace.t)

    if trace.legacy_mode is not None:
        legacy_mode = trace.legacy_mode.astype(bool)
    else:
        sums = window_sums(trace.decision.tolist(), legacy.n_bits)
        legacy_mode = np.array([legacy.mode(int(s)) for s in sums], dtype=bool)
    leg_rate, leg_share = _switch_rate(legacy_mode, trace.t)

    return TraceReport(
        n_checks=len(trace),
        duration_s=trace.duration,
        share_moves=float(trace.decision.mean()),
        mean_interval_s=float(trace.dt.mean()),
        naive_rate=est.naive,
        debiased_rate=est.debiased,
        cloglog=fit,
        controller_alarms=alarms,
        controller_switches_per_hour=ctrl_rate,
        controller_share_nonstationary=ctrl_share,
        legacy_switches_per_hour=leg_rate,
        legacy_share_nonstationary=leg_share,
    )
