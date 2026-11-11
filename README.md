# trosna-regime-control

[![PyPI](https://img.shields.io/pypi/v/trosna-regime-control.svg)](https://pypi.org/project/trosna-regime-control/)
[![Python](https://img.shields.io/pypi/pyversions/trosna-regime-control.svg)](https://pypi.org/project/trosna-regime-control/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/89605502155/trosna-regime-control/blob/main/LICENSE)

**Regime detection from reallocation decisions under irregular checks, scale-invariant monitoring intervals and simplex (Aitchison) memory allocation for adaptive KVM memory management.**

An adaptive memory manager for virtual machines checks the demand at intervals,
and moves memory by ballooning when the demand has changed. Each check leaves
one decision bit: 1 if memory was moved, 0 if not. trosna-regime-control turns this stream
of bits into the three things the manager needs:

* **the regime**: is the decision rate stationary or changing? It uses a
  Shiryaev–Roberts detector with a guaranteed mean time to a false alarm, and it
  works with irregular, policy-dependent intervals;
* **the next interval**: from a real-time estimate of the decision intensity
  with exponential forgetting, by the scale-invariant rules `Δt = p*/λ̂` and
  `Δt = √(2C₁/(λ̂C₂))`, not by a hand-made table;
* **the allocation**: memory shares as points of the simplex, perturbed in the
  clr coordinates, with minimum and maximum bounds and a dead band δ against
  needless reallocations.

It also diagnoses the 5-bit heuristic that it replaces: the exact flicker rate
of its regime, from a Markov chain, and a replay of both methods on a recorded
log.

Russian version: [README.ru.md](https://github.com/89605502155/trosna-regime-control/blob/main/README.ru.md).
Documentation: [English](https://github.com/89605502155/trosna-regime-control/tree/main/docs/en) ·
[Русский](https://github.com/89605502155/trosna-regime-control/tree/main/docs/ru) ·
[Changelog](https://github.com/89605502155/trosna-regime-control/blob/main/CHANGELOG.md) ·
[Issues](https://github.com/89605502155/trosna-regime-control/issues)

The library accompanies the article on Lie groups and martingales in adaptive
KVM memory management by Andrey Ferubko and Oleg Kazakov, Bryansk State
Engineering-Technological University (BGITU). The theory is in
[docs/en/theory.md](https://github.com/89605502155/trosna-regime-control/blob/main/docs/en/theory.md).

## Installation

```sh
pip install trosna-regime-control
# or, with uv
uv add trosna-regime-control
```

Requires Python 3.11 or newer and NumPy. The optional extra `plot` adds
matplotlib for the experiment scripts.

## Quick start

```python
from trosna_regime_control import ControllerConfig, RegimeController

controller = RegimeController(ControllerConfig(rule="A", p_star=0.3))
t = 0.0
interval = controller.start(t)  # seconds until the first check
for _ in range(20):
    t += interval
    moved = False  # the real decision of the system: True if memory was moved
    step = controller.record_decision(t, moved)
    interval = step.next_dt
print(f"next check in {step.next_dt:.1f} s, non-stationary: {step.nonstationary}")
```

On a recorded log (`checks.csv`, see [docs/en/log_format.md](https://github.com/89605502155/trosna-regime-control/blob/main/docs/en/log_format.md)):

```sh
trosna-regime-control analyze checks.csv --rule A --p-star 0.3
trosna-regime-control flicker --heuristic config/heuristic.json
```

## How it differs from existing packages

Compositional-data libraries, such as `scikit-bio` (`skbio.stats.composition`),
`composition_stats` and `pyCoDA`, implement the centred and isometric log-ratio
transforms. They do not allocate resources: none of them has bounds on the
shares, a dead band against needless reallocations or a monitoring interval
derived from the decisions. Change-point libraries, such as `ruptures`,
`ocpdet` and `changepoint-online`, assume regular sampling and, mostly,
Gaussian data. None of them handles Bernoulli decisions at irregular,
policy-dependent intervals with a guarantee on the false-alarm rate.

| Feature | trosna-regime-control | compositional-data libraries | change-point libraries |
|---|---|---|---|
| clr and softmax, Aitchison distance | yes | yes | no |
| Memory allocation on the simplex with bounds and a dead band δ | yes | no | no |
| Monitoring interval by scale-invariant rules, cost model `C₁/C₂` | yes | no | no |
| Real-time decision intensity with exponential forgetting and debiasing | yes | no | no |
| Shiryaev–Roberts and CUSUM on Bernoulli decisions at irregular intervals | yes | no | no, they assume regular sampling |
| ARL lower bound for adaptive intervals (mean time to a false alarm) | yes, under the model | no | no |
| Exact flicker of the 5-bit heuristic (Markov chain) | yes | no | no |
| Streaming controller with JSON state and CLI | yes | no | partly |

The package does not estimate the growth of memory with Azuma–Hoeffding
intervals or betting martingales. That is the task of `azarovo-martingale`.
trosna-regime-control is about the regime of the decision stream.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest            # fast tests; Monte Carlo ones are marked slow
uv run pytest -m slow    # Monte Carlo check of the ARL (minutes)
```

Commit messages in this repository have a subject of at most 8 characters and
no body. Install the hook once per clone with `scripts/install_hooks.sh`.

## Citation

See [CITATION.cff](https://github.com/89605502155/trosna-regime-control/blob/main/CITATION.cff)
or the "Cite this repository" button on GitHub.

## Authors

- Andrey Ferubko, [ORCID 0009-0006-5625-137X](https://orcid.org/0009-0006-5625-137X)
- Oleg Kazakov, [ORCID 0000-0001-9665-8138](https://orcid.org/0000-0001-9665-8138)

Bryansk State Engineering-Technological University (BGITU).

## License

MIT. Copyright (c) 2026 Andrey Ferubko, Oleg Kazakov. See [LICENSE](https://github.com/89605502155/trosna-regime-control/blob/main/LICENSE).
