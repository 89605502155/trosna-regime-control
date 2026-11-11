# Command line

[Русская версия](../ru/cli.md) · [README](../../README.md)

```sh
trosna-regime-control --version
trosna-regime-control analyze checks.csv [options]
trosna-regime-control flicker [--heuristic config.json] [--rate 0.05 ...] [--json]
```

The same tool runs as `python -m trosna_regime_control`. Exit code 0 means
success, 2 means a bad input, with the message on standard error.

## analyze

Reads a `checks.csv` file (format in [log_format.md](log_format.md)) and prints
the summary:

* the number of checks, the share of moves, the mean interval;
* the naive and debiased rates at the end of the trace;
* the model check: the slope of `ln Δt` (model: 1), the coefficient of the
  previous bit (model: 0), and whether the model is consistent;
* the proposed controller replayed on the same decisions: alarms, regime
  switches per hour, share of non-stationary checks;
* the current 5-bit heuristic on the same decisions: switches per hour and the
  share of non-stationary checks. The logged `mode` column is used if present.

| Option | Default | Meaning |
|---|---|---|
| `--tau` | 1800 | memory of the rate estimate, s |
| `--rule` | A | interval rule, `A` or `B` |
| `--p-star` | 0.3 | target share of moves for rule A |
| `--c1-over-c2` | 2.0 | ratio of costs for rule B, s |
| `--arl` | 3600 | mean time to a false alarm of the detector, s |
| `--hold` | 300 | time the non-stationary regime lasts after an alarm, s |
| `--warmup` | 600 | observation time before the detector is armed, s |
| `--heuristic` | built-in | JSON file with the table of the current heuristic |
| `--burn-in` | 10 | checks skipped by the model check |
| `--json` | off | print the report as JSON |

```sh
trosna-regime-control analyze checks.csv --rule B --c1-over-c2 2 --json
```

## flicker

Computes the exact switches per hour of the current heuristic at constant
rates, from the Markov chain. Without `--rate` it prints 0.01, 0.02, 0.05, 0.1
and 0.2 1/s. A note on standard error marks a conditional table
(`is_real` false).

```sh
trosna-regime-control flicker --heuristic config/heuristic.json --rate 0.05 --rate 0.1
```

## The heuristic configuration file

```json
{
  "comment": "parameters of the current heuristic",
  "is_real": true,
  "n_bits": 5,
  "interval_by_sum_s": {"0": 10.0, "1": 8.2, "2": 6.4, "3": 4.6, "4": 2.8, "5": 1.0},
  "nonstationary_if_sum_at_least": 3
}
```

`interval_by_sum_s` must give a value for every sum from 0 to `n_bits`.
`is_real` must be `true` for results that go into a paper.
