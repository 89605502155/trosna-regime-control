# API

[Русская версия](../ru/api.md) · [README](../../README.md)

All public names are importable from `trosna_regime_control`. Arrays are NumPy
arrays. Times are seconds, rates are 1/s, switches are per hour.

## Simplex and allocation (idea 1)

| Name | Purpose |
|---|---|
| `closure_shares(x)` | normalise positive weights to shares |
| `perturb(w, p)` | group operation `w ⊕ p` |
| `clr(w)`, `clr_inv(z)` | logarithm (centred log-ratio) and exponential (softmax) |
| `aitchison_distance(w1, w2)` | distance `‖clr(w1) − clr(w2)‖` |
| `plan_reallocation(current, target, delta)` | dead-band decision, returns `Reallocation` |
| `allocate(total, shares, m_min, m_max)` | memory per VM with bounds, returns `Allocation` |

```python
import numpy as np
from trosna_regime_control import allocate, clr, clr_inv, perturb

w = np.array([0.5, 0.3, 0.2])
p = np.array([1.2, 1.0, 1.0])
assert np.allclose(clr(perturb(w, p)), clr(w) + clr(p))
assert np.allclose(clr_inv(clr(w)), w)
print(allocate(1000.0, w, [100.0, 100.0, 100.0]).memory)
```

## Intervals (idea 2)

| Name | Purpose |
|---|---|
| `IntervalBounds(lo, hi)` | bounds of the interval, default 1 s … 10 s |
| `rule_a(rate, p_star)` | `Δt = p*/λ̂` |
| `rule_b(rate, c1_over_c2)` | `Δt = √(2C₁/(λ̂C₂))` |
| `rule_c(drift, delta)` | `Δt = δ/v` |
| `exact_interval(rate, c1_over_c2)` | exact cost optimum, bisection, no SciPy |

```python
from trosna_regime_control import IntervalBounds, exact_interval, rule_a, rule_b

bounds = IntervalBounds(lo=1.0, hi=10.0)
print(float(rule_a(0.05, 0.3, bounds)), round(float(rule_b(0.05, 2.0, bounds)), 2))
print(round(exact_interval(0.05, 2.0), 2))
```

## Intensity (idea 3)

`ForgettingRate(tau, prior_rate=None, prior_time=0.0, prior_dt=5.0, *, confidence=0.95, z=None)`
has:

* `record_check(decision, dt) -> float`: record one check, return the naive rate;
* `naive`, `debiased`: the estimates, 1/s (NaN before data);
* `upper_naive()`, `lower_naive()`: one-sided confidence bounds of the naive rate,
  with `z` standard deviations. `z` comes from `confidence` (default 0.95, so
  `z = 1.6449`), or is given directly as `z`;
* `z`: the number of standard deviations in use;
* `to_dict()`, `from_dict(data)`: state for storage.

`naive_bias(rate, dt)` gives the relative bias of the naive estimate.

## Detectors (idea 4)

| Name | Purpose |
|---|---|
| `lr_factor(decision, dt, lam0, kappa)` | likelihood-ratio factor `f` |
| `ShiryaevRoberts(lam0, kappa, threshold, time_weighted=True)` | the statistic `R`, alarm at `R ≥ threshold`, then reset; `record_check(decision, dt) -> bool` |
| `ShiryaevRoberts.for_arl(lam0, kappa, arl)` | threshold equal to the required ARL, in seconds |
| `Cusum(lam0, kappa, h)` | CUSUM statistic; `record_check(decision, dt) -> bool` |
| `cusum_threshold_for_arl(arl)` | `h = ln(ARL)` |
| `TwoSided(lam0_up, threshold, kappa_up, kappa_down, time_weighted, lam0_down)` | rise and fall; `record_check` returns `+1`, `−1` or `0` |
| `run_lengths(kind, threshold, lam_true, lam0, kappa, dt, n_runs, seed)` | Monte Carlo run lengths, vectorised |

Functions `lorden_delay_bound`, `kl_rate_poisson` and `kl_per_check` are in
`trosna_regime_control.detectors`.

```python
from trosna_regime_control import ShiryaevRoberts

det = ShiryaevRoberts.for_arl(0.05, 2.0, arl=3600.0)
print(det.record_check(1, 10.0), round(det.stat, 3))
```

## Flicker of the current heuristic (idea 3)

| Name | Purpose |
|---|---|
| `flicker_stats(rate, config=None)` | exact switches per hour, share non-stationary, mean interval |
| `switches_per_hour(decisions, dt, config, burn_in)` | the same on a recorded sequence |
| `HeuristicConfig(table, threshold, n_bits, is_real)` | table of the 5-bit rule |
| `load_heuristic_config(path)` | read the JSON table |
| `default_config()`, `DEFAULT_TABLE` | the conditional table of the document |
| `window_sums(bits)` | sum `S` after each check |
| `LegacyHeuristic(config)` | the 5-bit rule as a scheduler |

```python
from trosna_regime_control import flicker_stats

print(round(flicker_stats(0.05).switches_per_hour, 1))
```

## Controller

`ControllerConfig` has the fields `tau_s`, `rule` (`"A"` or `"B"`), `p_star`,
`c1_over_c2`, `bounds`, `arl_s`, `kappa_up`, `kappa_down`, `hold_s`,
`warmup_s`, `tau_fast_s`, `min_rate`, `confidence` (default 0.95), `z`
(default `None`), `prior_rate`, `prior_time_s`. See the docstring of each field.
`z_effective` is the number of standard deviations in use.

`RegimeController(config=None, *, confidence=None, z=None)`. The keyword
arguments replace the confidence level of `config` (give one of them, not both):

* `start(t0) -> float`: begin, return the first interval;
* `record_decision(t, decision) -> Step`: record a check, validated by
  `validate_decision`;
* `baselines`, `rate`, `n_checks`, `n_alarms`: state;
* `to_dict()`, `from_dict(data)`, `to_json()`, `from_json(text)`: save and restore.
  The state version is 2. A version 1 state is refused.

`Step` has `t_next`, `next_dt`, `nonstationary`, `alarm` (`"rise"`, `"fall"`
or `None`) and `rate`.

## Schedulers and simulation

`Scheduler` is the protocol with `start` and `record_decision`. `FixedInterval(dt)`,
`LegacyHeuristic`, `UniformIntervals` and `RegimeController` implement it. `replay(scheduler, trace)`
feeds a recorded trace. `simulate(scheduler, StepIntensity(...), horizon_s, seed,
carryover)` runs the loop on a Poisson demand. `UniformIntervals(lo, hi, seed)`
draws random intervals, which the model check needs.

## Traces and analysis

`read_checks(path) -> Trace` reads `checks.csv`. `Trace` has the arrays `t`,
`dt`, `decision` and the optional `legacy_mode` and `s_sum`.
`analyze(trace, controller_config, legacy_config, burn_in) -> TraceReport`
returns the summary. `fit_cloglog(decisions, dt, burn_in) -> CloglogFit` checks
the model. `ones_share_by_interval` gives the share of moves per interval group.

## Validation and normal quantiles

`validate_decision(t, decision, last_t=None) -> (float, bool)` checks one check.
The time must be finite, non-negative and later than `last_t`. The decision must
be a bool, a numpy bool, or the integer 0 or 1. Floats are refused, because a
fractional decision is not a Bernoulli outcome.

`z_for_confidence(c)` gives `Φ⁻¹(c)` for a one-sided level `0.5 ≤ c < 1`, and
`norm_ppf(p)` is the standard normal quantile (Wichura's AS 241, no SciPy).

## Naming

Every public method that takes one observation and updates the state is called
`record_*`: `record_decision` on the controller and the schedulers, and
`record_check` on the estimator and the detectors. The name `update` is not used
for these methods, because it did not say what they do. The rename of the
unpublished draft has no deprecated aliases. See [migration.md](migration.md).

## Errors

Invalid parameters raise `ValueError` with a message, and a wrong type raises
`TypeError`. Calling `record_decision` before `start` raises `RuntimeError`.
A file that does not match the format raises `ValueError` from `read_checks`.
