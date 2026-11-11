# Changelog

All notable changes of trosna-regime-control are documented here. The project
follows [Semantic Versioning](https://semver.org). Russian summary: the same
entries apply; the documentation is in `docs/ru/`.

## [0.0.1] — 2026-11-11

First release: the methods of the article on Lie groups and martingales in
adaptive KVM memory management (A. Ferubko, O. D. Kazakov, BGITU). The draft that
preceded it was never published; the changes against that draft are listed at the end.

* Memory shares on the simplex: `perturb`, `clr`, `clr_inv`,
  `aitchison_distance`, the dead-band planner `plan_reallocation` and
  `allocate` with minimum and maximum bounds.
* Monitoring intervals by scale-invariant rules A and B, the cost rate, the
  exact optimum `exact_interval` (no SciPy) and rule C.
* Real-time decision intensity with exponential forgetting: `ForgettingRate`
  with the naive and debiased estimates and confidence bounds.
* Martingale detectors on Bernoulli decisions at irregular intervals:
  Shiryaev–Roberts (count and time weighted), CUSUM, two-sided `TwoSided`,
  `run_lengths` for Monte Carlo.
* Exact flicker of the current 5-bit heuristic from a Markov chain
  (`flicker_stats`), `LegacyHeuristic` and the JSON configuration.
* Model check of `P(b = 1) = 1 − e^{−λΔt}` by a cloglog fit in numpy
  (`fit_cloglog`), with the slope-fixed rate.
* Streaming `RegimeController` with JSON state, `replay` and `simulate`.
* Command-line tool `trosna-regime-control` with `analyze` and `flicker`.
* Documentation in English and Russian, five examples, tests (fast, with a
  Monte Carlo case marked `slow`), the heavy experiment script `arl_study.py`.
* Baseline confidence of the detectors: `confidence` (one-sided, default 0.95) or
  `z` in `ControllerConfig`, `RegimeController(…, confidence=…, z=…)` and
  `ForgettingRate`. The quantile `z_for_confidence` (Wichura AS 241, no SciPy).
* `validate_decision`: time and decision checked before a check enters the controller.
* JSON state version 2.
* Command-line options `--confidence` and `--z` for `analyze`.

Changes against the unpublished draft:

* Streaming methods renamed to `record_*`: `RegimeController.record_decision`,
  `Scheduler.record_decision` (with `FixedInterval`, `LegacyHeuristic`,
  `UniformIntervals`) and `record_check` on `ForgettingRate`, `ShiryaevRoberts`,
  `Cusum` and `TwoSided`. The name `update` is no longer used. No aliases.
* `baseline_z = 1.645` replaced by `confidence = 0.95` (`z = 1.6449`).
* `upper_naive()` and `lower_naive()` take no argument: `z` is set in the constructor.
* Publishing is manual; the repository has no automated publishing workflow.
