import json

import numpy as np
import pytest

from trosna_regime_control import (
    DEFAULT_TABLE,
    HeuristicConfig,
    LegacyHeuristic,
    flicker_stats,
    load_heuristic_config,
    simulate,
    switches_per_hour,
    window_sums,
)
from trosna_regime_control.flicker import markov_chain
from trosna_regime_control.heuristic import default_config
from trosna_regime_control.synthetic import StepIntensity


def test_markov_chain_is_stochastic() -> None:
    mc = markov_chain(0.05)
    assert np.allclose(mc.transition.sum(axis=1), 1.0)
    assert np.allclose(mc.stationary @ mc.transition, mc.stationary, atol=1e-10)


def test_flicker_numbers_from_document() -> None:
    assert flicker_stats(0.05).switches_per_hour == pytest.approx(57.4, abs=0.1)
    assert flicker_stats(0.10).switches_per_hour == pytest.approx(137.2, abs=0.1)


def test_markov_matches_simulation() -> None:
    tr = simulate(LegacyHeuristic(), StepIntensity(0.05), horizon_s=60_000.0, seed=21)
    sim = switches_per_hour(tr.decision, tr.dt, burn_in=100)
    assert sim == pytest.approx(flicker_stats(0.05).switches_per_hour, rel=0.12)


def test_window_sums() -> None:
    assert window_sums([1, 1, 0, 0, 0, 0, 1]).tolist() == [1, 2, 2, 2, 2, 1, 1]


def test_legacy_scheduler_follows_table() -> None:
    legacy = LegacyHeuristic()
    assert legacy.start(0.0) == DEFAULT_TABLE[0]
    step = legacy.update(10.0, True)
    assert step.next_dt == DEFAULT_TABLE[1]
    assert step.t_next == pytest.approx(10.0 + DEFAULT_TABLE[1])


def test_config_roundtrip_and_validation(tmp_path) -> None:
    data = {
        "is_real": True,
        "n_bits": 5,
        "interval_by_sum_s": {str(s): float(10 - s) for s in range(6)},
        "nonstationary_if_sum_at_least": 2,
    }
    path = tmp_path / "h.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    loaded = load_heuristic_config(path)
    assert loaded.table == (10.0, 9.0, 8.0, 7.0, 6.0, 5.0)
    assert loaded.threshold == 2
    assert loaded.is_real
    with pytest.raises(ValueError, match="n_bits"):
        HeuristicConfig((1.0, 2.0), 1)


def test_default_config_is_conditional() -> None:
    assert not default_config().is_real


def test_switches_need_enough_checks() -> None:
    with pytest.raises(ValueError, match="too few"):
        switches_per_hour([0, 1], [1.0, 1.0], burn_in=5)
