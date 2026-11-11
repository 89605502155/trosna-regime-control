# Quick start

[Русская версия](../ru/quickstart.md) · [README](../../README.md)

The examples below run as written. Each block runs in a temporary directory.

## 1. Intensity of the decisions in real time

A decision bit is 1 if memory was moved at a check. Each check comes with the
real interval since the previous one. `ForgettingRate` keeps three numbers and
forgets old checks at the rate of real time.

```python
from trosna_regime_control import ForgettingRate

est = ForgettingRate(tau=1800.0)  # memory of about 30 minutes
for moved, dt in [(0, 10.0), (1, 8.2), (0, 6.4), (1, 4.6)]:
    est.record_check(moved, dt)
print(f"naive {est.naive:.4f} 1/s, debiased {est.debiased:.4f} 1/s")
```

The naive rate `N/T` is biased low when `λΔt` is large. The debiased rate
inverts `P(b = 1) = 1 − e^{−λΔt}` and is the one to use.

## 2. The next interval from a rule

Rule A gives the interval `−ln(1 − p*)/λ̂` (the exact form): with the true rate,
the probability of a move in one interval is `p*`. The linear form `p*/λ̂` is
its first-order approximation and is shorter by about 19 % at `p* = 0.3`. Rule B minimises the cost
of checks plus delay. Both are clipped to `[1 s, 10 s]` by default.

```python
from trosna_regime_control import rule_a, rule_b

print(round(float(rule_a(0.05, p_star=0.3)), 2))  # 7.13 s, exact
print(round(float(rule_a(0.05, p_star=0.3, formula="linear")), 2))  # 6.0 s, linear
print(round(float(rule_b(0.05, c1_over_c2=2.0)), 2))  # 8.94 s
```

## 3. A detector of a change of regime

`ShiryaevRoberts.for_arl` sets the threshold to the required mean time to a
false alarm, in seconds. The statistic uses the real intervals.

```python
from trosna_regime_control import ShiryaevRoberts

det = ShiryaevRoberts.for_arl(lam0=0.05, kappa=2.0, arl=3600.0)
alarms = [det.record_check(decision, dt) for decision, dt in [(1, 5.0), (0, 5.0), (1, 5.0)]]
print(alarms, round(det.stat, 3))
```

## 4. The whole loop: the streaming controller

`RegimeController` combines the estimate, the detector and the rule. It is
started once and then fed every check. It returns the next interval and the regime.

```python
from trosna_regime_control import ControllerConfig, RegimeController

ctrl = RegimeController(ControllerConfig(rule="A", p_star=0.3))
t = 0.0
due = ctrl.start(t)
for moved in [False, True, False, False, True]:
    t += due  # in a real system: the time of the check that has just been made
    step = ctrl.record_decision(t, moved)
    due = step.next_dt
print(f"next check in {step.next_dt:.2f} s, alarm {step.alarm}")
```

The state is plain JSON. Save it with `ctrl.to_json()` and restore it with
`RegimeController.from_json(text)`.

## 5. Memory shares on the simplex

Shares are positive and sum to one. The group operation is `perturb`, and
`clr` is its logarithm. `plan` moves memory only if the target is farther than
the dead band `delta`.

```python
from trosna_regime_control import allocate, plan_reallocation

decision = plan_reallocation([0.5, 0.5], [0.7, 0.3], delta=0.05)
print(decision.reallocate, [round(x, 2) for x in decision.shares])
alloc = allocate(4096.0, decision.shares, m_min=[512.0, 512.0], m_max=[3584.0, 3584.0])
print([round(x, 1) for x in alloc.memory])
```

## 6. A recorded log

```python
import pathlib

from trosna_regime_control import analyze, read_checks

path = pathlib.Path("checks.csv")
path.write_text(
    "t_unix,dt_s,b\n"
    "1763200000.0,10.0,0\n"
    "1763200010.0,10.0,1\n"
    "1763200018.2,8.2,0\n"
    "1763200026.4,8.2,0\n"
    "1763200034.6,8.2,1\n",
    encoding="utf-8",
)
report = analyze(read_checks(path))
print(report.n_checks, round(report.share_moves, 2))
```

From the command line the same report is `trosna-regime-control analyze checks.csv`.
See [cli.md](cli.md) for the options and [log_format.md](log_format.md) for the columns.

## Next steps

* [theory.md](theory.md): the ideas in plain words, with the formulas.
* [api.md](api.md): all public names.
* [migration.md](migration.md): how to move from the 5-bit heuristic.
