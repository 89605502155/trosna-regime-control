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

`ForgettingRate(tau, prior_rate=None, prior_time=0.0, prior_dt=5.0)` has:

* `update(decision, dt) -> float`: account for one check, return the naive rate;
* `naive`, `debiased`: the estimates, 1/s (NaN before data);
* `upper_naive(z)`, `lower_naive(z)`: confidence bounds of the naive rate;
* `to_dict()`, `from_dict(data)`: state for storage.

`naive_bias(rate, dt)` gives the relative bias of the naive estimate.

## Detectors (idea 4)

| Name | Purpose |
|---|---|
| `lr_factor(decision, dt, lam0, kappa)` | likelihood-ratio factor `f` |
| `ShiryaevRoberts(lam0, kappa, threshold, time_weighted=True)` | the statistic `R`, alarm at `R ≥ threshold`, then reset |
| `ShiryaevRoberts.for_arl(lam0, kappa, arl)` | threshold equal to the required ARL, in seconds |
| `Cusum(lam0, kappa, h)` | CUSUM statistic |
| `cusum_threshold_for_arl(arl)` | `h = ln(ARL)` |
| `TwoSided(lam0_up, threshold, kappa_up, kappa_down, time_weighted, lam0_down)` | rise and fall; `update` returns `+1`, `−1` or `0` |
| `run_lengths(kind, threshold, lam_true, lam0, kappa, dt, n_runs, seed)` | Monte Carlo run lengths, vectorised |

Functions `lorden_delay_bound`, `kl_rate_poisson` and `kl_per_check` are in
`trosna_regime_control.detectors`.

```python
from trosna_regime_control import ShiryaevRoberts

det = ShiryaevRoberts.for_arl(0.05, 2.0, arl=3600.0)
print(det.update(1, 10.0), round(det.stat, 3))
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
`warmup_s`, `tau_fast_s`, `min_rate`, `baseline_z`, `prior_rate`,
`prior_time_s`. See the docstring of each field.

`RegimeController(config)`:

* `start(t0) -> float`: begin, return the first interval;
* `update(t, decision) -> Step`: account for a check;
* `baselines`, `rate`, `n_checks`, `n_alarms`: state;
* `to_dict()`, `from_dict(data)`, `to_json()`, `from_json(text)`: save and restore.

`Step` has `t_next`, `next_dt`, `nonstationary`, `alarm` (`"rise"`, `"fall"`
or `None`) and `rate`.

## Schedulers and simulation

`Scheduler` is the protocol with `start` and `update`. `FixedInterval(dt)`,
`LegacyHeuristic` and `RegimeController` implement it. `replay(scheduler, trace)`
feeds a recorded trace. `simulate(scheduler, StepIntensity(...), horizon_s, seed,
carryover)` runs the loop on a Poisson demand. `UniformIntervals(lo, hi, seed)`
draws random intervals, which the model check needs.

## Traces and analysis

`read_checks(path) -> Trace` reads `checks.csv`. `Trace` has the arrays `t`,
`dt`, `decision` and the optional `legacy_mode` and `s_sum`.
`analyze(trace, controller_config, legacy_config, burn_in) -> TraceReport`
returns the summary. `fit_cloglog(decisions, dt, burn_in) -> CloglogFit` checks
the model. `ones_share_by_interval` gives the share of moves per interval group.

## Errors

Invalid parameters raise `ValueError` with a message. Calling `update` before
`start` raises `RuntimeError`. A file that does not match the format raises
`ValueError` from `read_checks`.
