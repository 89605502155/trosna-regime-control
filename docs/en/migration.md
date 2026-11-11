# Migration: from the 5-bit heuristic to λ̂ and a detector

[Русская версия](../ru/migration.md) · [README](../../README.md)

The current system writes one bit per check and sets the interval and the regime
from the sum `S` of the last five bits. The migration keeps the same log, so
the change can be made in three steps, each reversible.

## Step 1. Measure the current system

1. Set `is_real = true` in the heuristic configuration with the real table and
   threshold, then run `trosna-regime-control flicker`. This gives the switches
   per hour of the current rule at the observed rates.
2. Log `checks.csv` as in [log_format.md](log_format.md).
3. Run `trosna-regime-control analyze checks.csv`. Check that the model holds
   (`model consistent yes`). If it does not, the detector guarantee is only
   approximate, and the report says so.

## Step 2. Replay on the recorded log

Feed the same decisions to the new controller and compare with the logged
regime. Nothing is changed in the system yet.

```python
from trosna_regime_control import (
    ControllerConfig,
    LegacyHeuristic,
    RegimeController,
    flicker_stats,
    read_checks,
    replay,
)

import pathlib

pathlib.Path("checks.csv").write_text(  # a stand-in for the recorded log of the system
    "t_unix,dt_s,b\n1763200000.0,10.0,0\n1763200010.0,10.0,1\n1763200018.2,8.2,0\n",
    encoding="utf-8",
)
trace = read_checks("checks.csv")
new = replay(RegimeController(ControllerConfig(rule="A", p_star=0.3)), trace)
old = replay(LegacyHeuristic(), trace)
print(sum(s.nonstationary for s in new), sum(s.nonstationary for s in old))
print(round(flicker_stats(0.05).switches_per_hour, 1), "switches/h, current rule at 0.05 1/s")
```

On the log, the new regime should change far less often than the old one.

## Step 3. Replace the rule in the system

The mapping of the old parts to the new ones:

| Old | New | Notes |
|---|---|---|
| table `S → Δt` | `RegimeController` interval, rule A or B | `rule_a`, `rule_b` |
| `S ≥ 3` → non-stationary | `TwoSided` alarm, `hold_s` | `arl_s` sets the false-alarm level |
| 5-bit history | `ForgettingRate` with `tau_s` | state is one dictionary |
| hand-made `1 s` after change | `bounds.lo` in non-stationary regime | |

Inside the monitoring loop:

```python
from trosna_regime_control import ControllerConfig, RegimeController

controller = RegimeController(ControllerConfig(rule="A", p_star=0.3, arl_s=3600.0))
interval = controller.start(0.0)


def on_check(now: float, moved: bool) -> float:
    """Return the interval to the next check."""
    global interval
    step = controller.update(now, moved)
    interval = step.next_dt
    return interval
```

Save the state between restarts with `controller.to_json()` and restore it with
`RegimeController.from_json(text)`.

## What changes in the results

* The interval depends on the estimated rate and no longer on the table.
* The regime changes only on an alarm, not on each check. Its frequency is
  set by `arl_s`.
* Rules A and B need the decision rate `λ̂`. Its memory `tau_s` should be a few
  times the longest interval that matters, about 30 minutes in the document.

## Check before the switch

* The trace of the new controller on the log stays within the bounds.
* The model check is consistent on the stationary part of the log.
* The number of alarms before a known load change is about zero for the chosen
  `arl_s`. Use `run_lengths` to see the false-alarm level for your threshold.
