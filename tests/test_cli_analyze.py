import json

import pytest

from trosna_regime_control import (
    ControllerConfig,
    StepIntensity,
    UniformIntervals,
    analyze,
    simulate,
)
from trosna_regime_control.cli import main
from trosna_regime_control.trace import Trace


@pytest.fixture
def checks_csv(tmp_path):
    tr = simulate(UniformIntervals(1.0, 10.0, seed=5), StepIntensity(0.05), 4000.0, seed=51)
    path = tmp_path / "checks.csv"
    lines = ["t_unix,dt_s,b"]
    base = 1_763_200_000.0
    for t, dt, b in zip(tr.t, tr.dt, tr.decision, strict=True):
        lines.append(f"{base + t:.3f},{dt:.3f},{int(b)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_analyze_text(checks_csv, capsys) -> None:
    assert main(["analyze", str(checks_csv), "--warmup", "100"]) == 0
    out = capsys.readouterr().out
    assert "checks" in out
    assert "model consistent" in out
    assert "controller switches/h" in out


def test_analyze_json(checks_csv, capsys) -> None:
    assert main(["analyze", str(checks_csv), "--json", "--rule", "B"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_checks"] > 300
    assert data["cloglog"]["n_obs"] > 300
    assert 0.0 < data["share_moves"] < 1.0


def test_analyze_api_matches_model_on_stationary_trace() -> None:
    tr = simulate(UniformIntervals(1.0, 10.0, seed=7), StepIntensity(0.04), 150_000.0, seed=55)
    rep = analyze(tr, ControllerConfig(warmup_s=3000.0))
    assert rep.debiased_rate == pytest.approx(0.04, rel=0.25)
    assert rep.cloglog is not None
    assert rep.cloglog.model_consistent


def test_analyze_uses_logged_legacy_mode(tmp_path, capsys) -> None:
    path = tmp_path / "checks.csv"
    path.write_text(
        "t_unix,dt_s,b,mode\n100,10,0,0\n110,10,1,0\n120,10,1,1\n130,10,0,1\n140,10,0,0\n",
        encoding="utf-8",
    )
    assert main(["analyze", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["legacy_switches_per_hour"] > 0.0


def test_analyze_bad_file_returns_error_code(tmp_path, capsys) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    assert main(["analyze", str(path)]) == 2
    assert "error" in capsys.readouterr().err


def test_flicker_table_and_json(capsys) -> None:
    assert main(["flicker", "--rate", "0.05", "--rate", "0.1"]) == 0
    out = capsys.readouterr()
    assert "57.45" in out.out
    assert "137.18" in out.out
    assert "conditional" in out.err
    assert main(["flicker", "--rate", "0.05", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["rate"] == 0.05


def test_flicker_with_config_file(tmp_path, capsys) -> None:
    cfg = {
        "is_real": True,
        "n_bits": 5,
        "interval_by_sum_s": {str(s): 10.0 - 1.5 * s for s in range(6)},
        "nonstationary_if_sum_at_least": 3,
    }
    path = tmp_path / "heuristic.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert main(["flicker", "--heuristic", str(path), "--rate", "0.05"]) == 0
    assert "conditional" not in capsys.readouterr().err


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.0.1" in capsys.readouterr().out


def test_trace_type_is_exported() -> None:
    assert Trace.__name__ == "Trace"


def test_confidence_options(checks_csv, capsys):
    assert main(["analyze", str(checks_csv), "--confidence", "0.99", "--json"]) == 0
    assert main(["analyze", str(checks_csv), "--z", "1.0"]) == 0
    capsys.readouterr()
    assert main(["analyze", str(checks_csv), "--confidence", "1.2"]) == 2
    assert "confidence" in capsys.readouterr().err
