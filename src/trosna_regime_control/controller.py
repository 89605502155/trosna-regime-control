"""Streaming regime controller: rate estimate, change detector and interval rule.

For every check the controller receives the time and the decision bit, and
returns the time of the next check and the regime. The loop is:

1. the forgetting estimate ``λ̂`` (:class:`~trosna_regime_control.intensity.ForgettingRate`)
   takes the check with its real interval (idea 3);
2. a two-sided time-weighted Shiryaev–Roberts detector compares the decisions
   with the baseline ``λ₀`` (idea 4). Its mean time to a false alarm is at least
   ``arl_s`` seconds, under the Poisson model. The baseline is the upper (rise)
   or lower (fall) confidence bound of the naive rate at the confidence level
   ``confidence`` (default 0.95, one-sided). After an alarm the baseline is the
   fast estimate, the detector restarts, and the regime is non-stationary for
   ``hold_s`` seconds;
3. in the stationary regime the interval is ``−ln(1 − p*)/λ̂`` (rule A, exact
   formula by default; ``p*/λ̂`` with ``formula="linear"``) or the
   square-root law (rule B) (idea 2). The rate ``λ̂`` is the debiased estimate by
   default, or the naive ``N/T`` with ``estimator="naive"``. In the non-stationary
   regime the interval is the smallest bound, so the system relearns quickly.

The state is a plain dictionary; :meth:`RegimeController.to_json` and
:meth:`RegimeController.from_json` save and restore it, so that a daemon can
restart without losing the statistics.

Examples
--------
>>> ctrl = RegimeController()
>>> ctrl.start(0.0)
10.0
>>> step = ctrl.record_decision(10.0, decision=True)
>>> step.nonstationary, step.alarm
(False, None)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .detectors import TwoSided
from .intensity import ForgettingRate, check_estimator
from .intervals import IntervalBounds, check_rule_a, rule_a, rule_b
from .normal import baseline_z, check_z, z_for_confidence
from .scheduler import Step
from .validation import validate_decision

STATE_VERSION = 2


@dataclass(frozen=True)
class ControllerConfig:
    """Parameters of :class:`RegimeController`. All times are in seconds.

    Attributes
    ----------
    tau_s : float
        Memory of the rate estimate that feeds the interval rule.
    rule : {"A", "B"}
        Interval rule: ``−ln(1 − p*)/λ̂`` (rule A, default) or the square-root law (rule B).
    p_star : float
        Target probability of a move in one interval, in ``(0, 1)`` (rule A).
    formula : {"exact", "linear"}
        Form of rule A: ``"exact"`` gives ``Δt = −ln(1 − p*)/λ̂`` (default),
        ``"linear"`` gives ``Δt = p*/λ̂``. Ignored by rule B.
    scale : float
        Positive multiplier of the rule A interval. Default 1. Ignored by rule B.
    estimator : {"naive", "debiased"}
        Rate ``λ̂`` used by the interval rule and by the re-baseline after an alarm:
        ``"naive"`` is ``N/T``, ``"debiased"`` is ``−ln(1 − N/K)/(T/K)`` (default).
        The pair ``("exact", "debiased")`` gives the steady share ``p*``, see
        :func:`~trosna_regime_control.intervals.steady_state_share`. The detector
        baselines always use the count bounds of the naive rate.
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
    confidence : float
        One-sided confidence level of the baseline bounds, ``0.5 ≤ confidence < 1``.
        The baseline is the upper bound (rise) or the lower bound (fall) of the
        naive rate at this level. Default 0.95. A higher level gives a higher rise
        baseline and a lower fall baseline, so fewer alarms.
    z : float, optional
        Standard deviations of the bounds, used directly. If given, it replaces
        ``confidence`` (which is still checked). Non-negative and finite.
    prior_rate, prior_time_s : float, optional
        Prior of the rate estimate, 1/s and seconds of pseudo-data.

    Examples
    --------
    >>> ControllerConfig(rule="B", c1_over_c2=1.5).rule
    'B'
    >>> round(ControllerConfig(confidence=0.99).z_effective, 4)
    2.3263
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
    confidence: float = 0.95
    z: float | None = None
    prior_rate: float | None = None
    prior_time_s: float = 0.0
    formula: Literal["exact", "linear"] = "exact"
    scale: float = 1.0
    estimator: Literal["naive", "debiased"] = "debiased"

    def __post_init__(self) -> None:
        """Check the parameters."""
        if self.rule not in ("A", "B"):
            msg = "rule must be 'A' or 'B'"
            raise ValueError(msg)
        if self.tau_s <= 0.0 or self.tau_fast_s <= 0.0:
            msg = "memory times must be positive"
            raise ValueError(msg)
        check_rule_a(self.p_star, self.formula, self.scale)
        check_estimator(self.estimator)
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
        z_for_confidence(self.confidence)  # raises for a level outside [0.5, 1)
        if self.z is not None:
            check_z(self.z)

    @property
    def z_effective(self) -> float:
        """Standard deviations of the baseline bounds: ``z`` if given, else from ``confidence``."""
        return baseline_z(self.confidence, self.z)


class RegimeController:
    """Streaming controller: ``start`` once, then ``record_decision`` on every check.

    Parameters
    ----------
    config : ControllerConfig, optional
        The parameters. Default: ``ControllerConfig()``.
    confidence : float, optional
        One-sided confidence level of the baseline bounds. If given, it replaces
        ``config.confidence`` and ``config.z`` (the level is then the source of z).
    z : float, optional
        Standard deviations of the baseline bounds. If given, it replaces
        ``config.confidence`` and ``config.z``. Cannot be combined with
        ``confidence``.

    Attributes
    ----------
    config : ControllerConfig
        The effective parameters, including the confidence level.
    n_checks : int
        Number of checks seen.
    n_alarms : int
        Number of detector alarms.

    Raises
    ------
    ValueError
        If both ``confidence`` and ``z`` are given, or a value is invalid.

    Examples
    --------
    >>> ctrl = RegimeController(confidence=0.99)
    >>> round(ctrl.config.z_effective, 4)
    2.3263
    """

    def __init__(
        self,
        config: ControllerConfig | None = None,
        *,
        confidence: float | None = None,
        z: float | None = None,
    ) -> None:
        if confidence is not None and z is not None:
            msg = "give either confidence or z, not both"
            raise ValueError(msg)
        base = config if config is not None else ControllerConfig()
        if z is not None:
            base = replace(base, z=z)
        elif confidence is not None:
            base = replace(base, confidence=confidence, z=None)
        self.config = base
        self.est = ForgettingRate(
            self.config.tau_s,
            prior_rate=self.config.prior_rate,
            prior_time=self.config.prior_time_s,
            confidence=self.config.confidence,
            z=self.config.z,
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
        """Intensity estimate used by the rule, 1/s: ``config.estimator`` (NaN before data)."""
        return self.est.estimate(self.config.estimator)

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
        self.next_dt = self._plan(self.rate)
        return self.next_dt

    def record_decision(self, t: float, decision: bool) -> Step:
        """Record the check made at time ``t`` and return the plan for the next one.

        Parameters
        ----------
        t : float
            Time of the check, seconds, later than the previous check.
        decision : bool
            True (or 1) if memory was moved at this check; False (or 0) if not.
            See :func:`~trosna_regime_control.validation.validate_decision`.

        Returns
        -------
        Step
            Time and interval of the next check, the regime and any alarm.

        Raises
        ------
        RuntimeError
            If :meth:`start` was not called.
        TypeError
            If ``t`` or ``decision`` has a wrong type.
        ValueError
            If ``t`` is NaN, infinite, negative or does not increase, or ``decision``
            is an integer other than 0 or 1.
        """
        if not self.started:
            msg = "call start() before record_decision()"
            raise RuntimeError(msg)
        time, moved = validate_decision(t, decision, last_t=self.last_t)
        dt = time - self.last_t
        bit = 1.0 if moved else 0.0
        self.est.record_check(bit, dt)
        self.fast.record_check(bit, dt)
        self.n_checks += 1

        alarm: str | None = None
        if self.detector is None and self._armed(time):
            armed = self.est.upper_naive()
            if math.isfinite(armed):
                self.detector = self._make_detector(armed, self.est.lower_naive())
        if self.detector is not None:
            verdict = self.detector.record_check(bit, dt)
            if verdict != 0:
                alarm = "rise" if verdict > 0 else "fall"
                self.n_alarms += 1
                self.nonstationary_until = time + self.config.hold_s
                fresh = self.fast.estimate(self.config.estimator)
                base = fresh if math.isfinite(fresh) else self.detector.lam0_up
                self.detector = self._make_detector(base, base)

        nonstationary = self.nonstationary_until is not None and time < self.nonstationary_until
        rate = self.rate
        next_dt = self.config.bounds.lo if nonstationary else self._plan(rate)
        self.last_t = time
        self.next_dt = next_dt
        return Step(
            t_next=time + next_dt,
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
            value = rule_a(
                rate,
                cfg.p_star,
                cfg.bounds,
                fallback=cfg.bounds.hi,
                formula=cfg.formula,
                scale=cfg.scale,
            )
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
        # States saved before the fields `formula` and `scale` existed used the
        # linear rule A with scale 1. States saved before `estimator` used the
        # debiased rate. Restoring them keeps that behaviour.
        cfg_data.setdefault("formula", "linear")
        cfg_data.setdefault("scale", 1.0)
        cfg_data.setdefault("estimator", "debiased")
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
