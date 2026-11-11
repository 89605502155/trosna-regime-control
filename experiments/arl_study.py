"""HEAVY: Monte Carlo of ARL and detection delay for the change detectors.

Not in the test suite and not run during development: each cell simulates thousands of
runs of up to ``max_steps`` checks. Expect hours on a laptop for the full grid.

Output: ``out/arl_study/arl_delay.csv`` (one row per threshold and multiplier)
and, if matplotlib is installed, ``out/arl_study/delay_vs_arl.png``.

Usage: python experiments/arl_study.py --runs 2000 --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from trosna_regime_control.detectors import run_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2000, help="runs per cell")
    parser.add_argument("--lam0", type=float, default=0.05, help="baseline rate, 1/s")
    parser.add_argument("--dt", type=float, default=1.0, help="constant interval, s")
    parser.add_argument("--quick", action="store_true", help="a small grid for a smoke test")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    out_dir = Path("out") / "arl_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [300.0, 1800.0] if args.quick else [300.0, 1800.0, 3600.0, 28800.0]
    kappas = [2.0] if args.quick else [2.0, 4.0]
    max_steps = 50_000 if args.quick else 2_000_000

    rows: list[dict[str, float | str]] = []
    for kind in ("sr", "cusum"):
        for arl in thresholds:
            threshold = arl if kind == "sr" else math.log(arl)
            false = run_lengths(
                kind,
                threshold,
                args.lam0,
                args.lam0,
                2.0,
                dt=args.dt,
                n_runs=args.runs,
                max_steps=max_steps,
                seed=args.seed,
            )
            arl_hat = float(np.mean(np.where(np.isfinite(false), false, max_steps)))
            for kappa in kappas:
                delay = run_lengths(
                    kind,
                    threshold,
                    kappa * args.lam0,
                    args.lam0,
                    kappa,
                    dt=args.dt,
                    n_runs=args.runs,
                    max_steps=max_steps,
                    seed=args.seed + 1,
                )
                finite = delay[np.isfinite(delay)]
                rows.append(
                    {
                        "kind": kind,
                        "threshold": threshold,
                        "kappa": kappa,
                        "arl_checks_mc": arl_hat,
                        "delay_checks_mean": float(finite.mean()) if finite.size else math.nan,
                        "delay_seconds_mean": float(finite.mean() * args.dt)
                        if finite.size
                        else math.nan,
                        "share_no_alarm": float(1.0 - finite.size / delay.size),
                    }
                )
                print(json.dumps(rows[-1]))

    path = out_dir / "arl_delay.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for kind in ("sr", "cusum"):
        for kappa in kappas:
            pts = [r for r in rows if r["kind"] == kind and r["kappa"] == kappa]
            ax.plot(
                [float(r["arl_checks_mc"]) for r in pts],
                [float(r["delay_seconds_mean"]) for r in pts],
                marker="o",
                label=f"{kind}, kappa = {kappa:g}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("ARL, checks")
    ax.set_ylabel("mean delay, s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delay_vs_arl.png", dpi=200)


if __name__ == "__main__":
    main()
