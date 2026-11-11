"""Regime detection, monitoring intervals and simplex allocation for KVM memory.

Regime detection from reallocation decisions under irregular checks,
scale-invariant monitoring intervals and simplex (Aitchison) memory allocation
for adaptive KVM memory management.

The library implements the four ideas of the article on Lie groups and
martingales for memory management (document ``idei_gruppy_li_martingaly.md``):

* :mod:`trosna_regime_control.simplex` — memory shares as points of the simplex, with the
  perturbation group, clr/softmax, the Aitchison distance, a dead band δ and
  allocation with bounds;
* :mod:`trosna_regime_control.intervals` — monitoring intervals from scale-invariant
  rules A, B and C;
* :mod:`trosna_regime_control.intensity` and :mod:`trosna_regime_control.flicker` —
  real-time decision intensity with exponential forgetting, and the exact flicker of the current
  5-bit heuristic;
* :mod:`trosna_regime_control.detectors` — Shiryaev–Roberts and CUSUM detectors on Bernoulli
  decisions at irregular intervals, with an ARL guarantee.

:class:`~trosna_regime_control.controller.RegimeController` combines them in a streaming
loop. Its focus is the regime of the decision stream, not the memory growth
itself (that is covered by ``azarovo-martingale``).
"""

from .analyze import TraceReport, analyze
from .controller import ControllerConfig, RegimeController
from .detectors import (
    Cusum,
    ShiryaevRoberts,
    TwoSided,
    cusum_threshold_for_arl,
    lr_factor,
    run_lengths,
)
from .flicker import FlickerStats, flicker_stats, switches_per_hour
from .heuristic import (
    DEFAULT_TABLE,
    HeuristicConfig,
    LegacyHeuristic,
    default_config,
    load_heuristic_config,
    window_sums,
)
from .intensity import ForgettingRate, naive_bias
from .intervals import IntervalBounds, exact_interval, rule_a, rule_b, rule_c
from .regression import CloglogFit, Coefficient, fit_cloglog, ones_share_by_interval
from .scheduler import FixedInterval, Scheduler, Step, replay
from .simplex import Allocation, Reallocation, aitchison_distance, allocate, clr, clr_inv, perturb
from .simplex import closure as closure_shares
from .simplex import plan as plan_reallocation
from .synthetic import StepIntensity, UniformIntervals, simulate
from .trace import Trace, read_checks

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TABLE",
    "Allocation",
    "CloglogFit",
    "Coefficient",
    "ControllerConfig",
    "Cusum",
    "FixedInterval",
    "FlickerStats",
    "ForgettingRate",
    "HeuristicConfig",
    "IntervalBounds",
    "LegacyHeuristic",
    "Reallocation",
    "RegimeController",
    "Scheduler",
    "ShiryaevRoberts",
    "Step",
    "StepIntensity",
    "Trace",
    "TraceReport",
    "TwoSided",
    "UniformIntervals",
    "aitchison_distance",
    "allocate",
    "analyze",
    "closure_shares",
    "clr",
    "clr_inv",
    "cusum_threshold_for_arl",
    "default_config",
    "exact_interval",
    "fit_cloglog",
    "flicker_stats",
    "load_heuristic_config",
    "lr_factor",
    "naive_bias",
    "ones_share_by_interval",
    "perturb",
    "plan_reallocation",
    "read_checks",
    "replay",
    "rule_a",
    "rule_b",
    "rule_c",
    "run_lengths",
    "simulate",
    "switches_per_hour",
    "window_sums",
]
