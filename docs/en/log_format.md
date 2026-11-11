# Log format

[Русская версия](../ru/log_format.md) · [README](../../README.md)

One file `checks.csv` holds one row per check. Encoding UTF-8.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `t_unix` | float | yes | moment of the check, seconds since the Unix epoch |
| `b` | 0 or 1 | yes | 1 if memory was moved at this check |
| `dt_s` | float | no | real interval before this check, seconds |
| `s_sum` | int | no | sum `S` of the current heuristic after the check |
| `mode` | 0 or 1 | no | regime of the current heuristic, 1 non-stationary |
| `next_dt_s` | float | no | interval the system chose, seconds (not read) |
| `cycle_cost_s` | float | no | duration of the cycle, seconds (not read) |
| `moved_mb` | float | no | memory moved in the cycle, MiB (not read) |

Rules:

* If `dt_s` is absent, the interval is `t_unix` minus the previous `t_unix`,
  and the first row is dropped (its time is used as the start).
* `t_unix` must increase. `dt_s` must be positive. `b` must be 0 or 1.
* The bit `b` belongs to the interval before its check: the decision is taken
  on the statistics of that check. This matters for the model `P(b = 1) = 1 − e^{−λΔt}`.

Example:

```
t_unix,dt_s,b,s_sum,mode,next_dt_s,cycle_cost_s,moved_mb
1763200000.000,10.000,0,0,0,10.000,0.041,0
1763200010.002,10.002,1,1,0,8.200,0.057,256
1763200018.204,8.202,0,1,0,8.200,0.043,0
```

Reading in Python:

```python
import pathlib

from trosna_regime_control import read_checks

path = pathlib.Path("checks.csv")
path.write_text(
    "t_unix,dt_s,b\n1763200000.0,10.0,0\n1763200010.0,10.0,1\n",
    encoding="utf-8",
)
trace = read_checks(path)
print(trace.t, trace.dt, trace.decision)
```

Measurements for the memory of virtual machines come from the balloon
statistics of libvirt, for example `virsh domstats --balloon`. The per-VM file
`vm_stats.csv` in the research folder follows the same idea.
