# Changelog

All notable changes of trosna-regime-control are documented here. The project
follows [Semantic Versioning](https://semver.org). Russian summary: the same
entries apply; the documentation is in `docs/ru/`.

## 0.1.0 — 2026-11-11

First release: the methods of the article on Lie groups and martingales in
adaptive KVM memory management (A. Ferubko, BGITU).

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
