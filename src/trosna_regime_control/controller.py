"""Streaming regime controller: rate estimate, change detector and interval rule.

For every check the controller receives the time and the decision bit, and
returns the time of the next check and the regime. The loop is:

1. the forgetting estimate ``λ̂`` (:class:`~trosna_regime_control.intensity.ForgettingRate`)
   takes the check with its real interval (idea 3);
2. a two-sided time-weighted Shiryaev–Roberts detector compares the decisions
   with the baseline ``λ₀`` (idea 4). Its mean time to a false alarm is at least
   ``arl_s`` seconds, under the Poisson model. After an alarm the baseline is
   the fast estimate, the detector restarts, and the regime is non-stationary
   for ``hold_s`` seconds;
3. in the stationary regime the interval is ``p*/λ̂`` (rule A) or the
   square-root law (rule B) (idea 2). In the non-stationary regime it is the
   smallest bound, so the system relearns quickly.

The state is a plain dictionary; :meth:`RegimeController.to_json` and
:meth:`RegimeController.from_json` save and restore it, so that a daemon can
restart without losing the statistics.

Examples
--------
>>> ctrl = RegimeController()
>>> ctrl.start(0.0)
10.0
>>> step = ctrl.update(10.0, decision=True)
>>> step.nonstationary, step.alarm
(False, None)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .detectors import TwoSided
from .intensity import ForgettingRate
from .intervals import IntervalBounds, rule_a, rule_b
from .scheduler import Step

STATE_VERSION = 1


@dataclass(frozen=True)
class ControllerConfig:
    """Parameters of :class:`RegimeController`. All times are in seconds.

    Attributes
    ----------
    tau_s : float
        Memory of the rate estimate that feeds the interval rule.
    rule : {"A", "B"}
        Interval rule: ``p*/λ̂`` or the square-root law.
    p_star : float
        Target share of checks with a move (rule A).
    c1_over_c2 : float
        Ratio of the cost of a check to the cost of a second of delay, s (rule B).
    bounds : IntervalBounds
        Interval bounds, default 1 s … 10 s.
    arl_s : float
        Threshold of the detector: the mean time to a false alarm is at least this.
    kappa_up, kappa_down : float
        Alternatives for a rise (> 1) and a fall (< 1) of the rate.
    hold_s : float
        How long the non-stationary regime lasts after an alarm.
    warmup_s : float
        Observation time before the detector is armed.
    tau_fast_s : float
        Memory of the fast estimate used as the new baseline after an alarm.
    min_rate : float
        Floor of the baseline, 1/s, so that the detector never divides by zero.
    baseline_z : float
        Standard deviations of the upper confidence bound used as the baseline
        when the detector is armed (1.645: 95 %). A larger baseline is
        conservative for the false-alarm guarantee.
    prior_rate, prior_time_s : float, optional
        Prior of the rate estimate, 1/s and seconds of pseudo-data.

    Examples
    --------
    >>> ControllerConfig(rule="B", c1_over_c2=1.5).rule
    'B'
    """

    tau_s: float = 1800.0
    rule: Literal["A", "B"] = "A"
    p_star: float = 0.3
    c1_over_c2: float = 2.0
    bounds: IntervalBounds = field(default_factory=IntervalBounds)
    arl_s: float = 3600.0
    kappa_up: float = 2.0
    kappa_down: float = 0.5
    hold_s: float = 300.0
    warmup_s: float = 600.0
    tau_fast_s: float = 300.0
    min_rate: float = 1e-4
    baseline_z: float = 1.645
    prior_rate: float | None = None
    prior_time_s: float = 0.0

    def __post_init__(self) -> None:
        """Check the parameters."""
        if self.rule not in ("A", "B"):
            msg = "rule must be 'A' or 'B'"
            raise ValueError(msg)
        if self.tau_s <= 0.0 or self.tau_fast_s <= 0.0:
            msg = "memory times must be positive"
            raise ValueError(msg)
        if not 0.0 < self.p_star < 1.0:
            msg = "p_star must be in (0, 1)"
            raise ValueError(msg)
        if self.c1_over_c2 <= 0.0:
            msg = "c1_over_c2 must be positive"
            raise ValueError(msg)
        if self.arl_s <= 1.0:
            msg = "arl_s must exceed 1 second"
            raise ValueError(msg)
        if self.kappa_up <= 1.0 or not 0.0 < self.kappa_down < 1.0:
            msg = "kappa_up must exceed 1 and kappa_down must be in (0, 1)"
            raise ValueError(msg)
        if self.hold_s < 0.0 or self.warmup_s < 0.0 or self.prior_time_s < 0.0:
            msg = "hold_s, warmup_s and prior_time_s must be non-negative"
            raise ValueError(msg)
        if self.min_rate <= 0.0:
            msg = "min_rate must be positive"
            raise ValueError(msg)
        if self.baseline_z < 0.0:
            msg = "baseline_z must be non-negative"
            raise ValueError(msg)


class RegimeController:
    """Streaming controller: ``start`` once, then ``update`` on every check.

    Attributes
    ----------
    config : ControllerConfig
        The parameters.
    n_checks : int
        Number of checks seen.
    n_alarms : int
        Number of detector alarms.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config if config is not None else ControllerConfig()
        self.est = ForgettingRate(
            self.config.tau_s,
            prior_rate=self.config.prior_rate,
            prior_time=self.config.prior_time_s,
        )
        self.fast = ForgettingRate(self.config.tau_fast_s)
        self.detector: TwoSided | None = None
        self.t_start = 0.0
        self.last_t = 0.0
        self.next_dt = self.config.bounds.hi
        self.nonstationary_until: float | None = None
        self.n_checks = 0
        self.n_alarms = 0
        self.started = False

    @property
    def rate(self) -> float:
        """Debiased intensity estimate, 1/s (NaN before data)."""
        return self.est.debiased

    @property
    def baselines(self) -> tuple[float, float] | None:
        """Baselines ``(λ₀⁺, λ₀⁻)`` of the rise and fall detectors, or None if not armed."""
        if self.detector is None:
            return None
        return self.detector.lam0_up, self.detector.lam0_down

    def start(self, t0: float) -> float:
        """Begin observation at ``t0``; return the first interval, seconds."""
        self.t_start = float(t0)
        self.last_t = float(t0)
        self.started = True
        self.next_dt = self._plan(self.est.debiased)
        return self.next_dt

    def update(self, t: float, decision: bool) -> Step:
        """Account for the check at time ``t`` and return the plan for the next one.

        Parameters
        ----------
        t : float
            Time of the check, seconds, after the previous check.
        decision : bool
            True if memory was moved at this check.

        Raises
        ------
        RuntimeError
            If :meth:`start` was not called.
        ValueError
            If ``t`` does not increase.
        """
        if not self.started:
            msg = "call start() before update()"
            raise RuntimeError(msg)
        dt = float(t) - self.last_t
        if dt <= 0.0:
            msg = "check times must increase"
            raise ValueError(msg)
        bit = 1.0 if decision else 0.0
        self.est.update(bit, dt)
        self.fast.update(bit, dt)
        self.n_checks += 1

        alarm: str | None = None
        if self.detector is None and self._armed(t):
            armed = self.est.upper_naive(self.config.baseline_z)
            if math.isfinite(armed):
                self.detector = self._make_detector(
                    armed, self.est.lower_naive(self.config.baseline_z)
                )
        if self.detector is not None:
            verdict = self.detector.update(bit, dt)
            if verdict != 0:
                alarm = "rise" if verdict > 0 else "fall"
                self.n_alarms += 1
                self.nonstationary_until = float(t) + self.config.hold_s
                fresh = self.fast.debiased
                base = fresh if math.isfinite(fresh) else self.detector.lam0_up
                self.detector = self._make_detector(base, base)

        nonstationary = self.nonstationary_until is not None and t < self.nonstationary_until
        rate = self.est.debiased
        next_dt = self.config.bounds.lo if nonstationary else self._plan(rate)
        self.last_t = float(t)
        self.next_dt = next_dt
        return Step(
            t_next=float(t) + next_dt,
            next_dt=next_dt,
            nonstationary=nonstationary,
            alarm=alarm,
            rate=rate,
        )

    # --- internals -----------------------------------------------------------

    def _armed(self, t: float) -> bool:
        return float(t) - self.t_start >= self.config.warmup_s

    def _make_detector(self, up: float, down: float) -> TwoSided:
        floor = self.config.min_rate
        return TwoSided(
            max(floor, up),
            lam0_down=max(floor, down),
            threshold=self.config.arl_s,
            kappa_up=self.config.kappa_up,
            kappa_down=self.config.kappa_down,
            time_weighted=True,
        )

    def _plan(self, rate: float) -> float:
        cfg = self.config
        if cfg.rule == "A":
            value = rule_a(rate, cfg.p_star, cfg.bounds, fallback=cfg.bounds.hi)
        else:
            value = rule_b(rate, cfg.c1_over_c2, cfg.bounds, fallback=cfg.bounds.hi)
        return float(value)

    # --- state ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Complete state as a JSON-serialisable dictionary."""
        detector: dict[str, Any] | None = None
        if self.detector is not None:
            detector = {
                "lam0_up": self.detector.lam0_up,
                "lam0_down": self.detector.lam0_down,
                "up": self.detector.up.stat,
                "down": self.detector.down.stat,
            }
        return {
            "version": STATE_VERSION,
            "config": asdict(self.config),
            "started": self.started,
            "t_start": self.t_start,
            "last_t": self.last_t,
            "next_dt": self.next_dt,
            "n_checks": self.n_checks,
            "n_alarms": self.n_alarms,
            "nonstationary_until": self.nonstationary_until,
            "estimator": self.est.to_dict(),
            "fast": self.fast.to_dict(),
            "detector": detector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeController:
        """Restore a controller from :meth:`to_dict` output."""
        if int(data.get("version", 0)) != STATE_VERSION:
            msg = "unsupported state version"
            raise ValueError(msg)
        cfg_data = dict(data["config"])
        cfg_data["bounds"] = IntervalBounds(**cfg_data["bounds"])
        ctrl = cls(ControllerConfig(**cfg_data))
        ctrl.started = bool(data["started"])
        ctrl.t_start = float(data["t_start"])
        ctrl.last_t = float(data["last_t"])
        ctrl.next_dt = float(data["next_dt"])
        ctrl.n_checks = int(data["n_checks"])
        ctrl.n_alarms = int(data["n_alarms"])
        until = data["nonstationary_until"]
        ctrl.nonstationary_until = None if until is None else float(until)
        ctrl.est = ForgettingRate.from_dict(data["estimator"])
        ctrl.fast = ForgettingRate.from_dict(data["fast"])
        det = data["detector"]
        if det is not None:
            ctrl.detector = ctrl._make_detector(float(det["lam0_up"]), float(det["lam0_down"]))
            ctrl.detector.up.stat = float(det["up"])
            ctrl.detector.down.stat = float(det["down"])
        return ctrl

    def to_json(self) -> str:
        """State as a JSON text (UTF-8 compatible, sorted keys)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> RegimeController:
        """Restore a controller from the text written by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))
