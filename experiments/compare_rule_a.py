"""Comparison of the four combinations of rule A: formula x estimator.

For every combination, every target ``p*``, every rate and every seed, the
controller runs on a Poisson demand with a constant rate ``λ``. The detector is
off (``warmup_s`` is infinite), so the run measures the interval rule alone. The
share of moves and ``λΔt`` are taken after a burn-in of ``burn_in_tau`` memory
times. The analytic steady state comes from
:func:`trosna_regime_control.intervals.steady_state_share`.

Seeds: the development seeds and the held-out seeds are disjoint. The defaults
of the library were not chosen on the held-out seeds.

Output: a Markdown table on stdout, and with ``--csv`` one row per run.

HEAVY on a laptop: the full grid (10 targets, 3 rates, 6 seeds, 4 combinations,
720 runs of up to a few hundred thousand checks) takes hours. The ``--quick``
option is the small configuration used in the documentation (about 10 s).

Usage::

    python experiments/compare_rule_a.py --quick
    python experiments/compare_rule_a.py --csv out/compare_rule_a/runs.csv   # HEAVY
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trosna_regime_control import ControllerConfig, RegimeController, StepIntensity, simulate
from trosna_regime_control.intervals import IntervalBounds, steady_state_share

FORMULAS = ("exact", "linear")
ESTIMATORS = ("naive", "debiased")
FULL_P_STAR = tuple(round(0.05 * k, 2) for k in range(1, 11))
FULL_RATES = (0.01, 0.05, 0.2)
DEV_SEEDS = (1, 2, 3)
HELDOUT_SEEDS = (101, 102, 103)


@dataclass(frozen=True)
class RunRow:
    """One run of one combination."""

    formula: str
    estimator: str
    p_star: float
    rate: float
    seed: int
    heldout: bool
    checks: int
    share: float
    lam_dt: float


def run_one(
    formula: str,
    estimator: str,
    p_star: float,
    rate: float,
    seed: int,
    *,
    horizon_s: float,
    tau_s: float,
    burn_in_tau: float,
    bounds: IntervalBounds,
) -> tuple[int, float, float]:
    """Return ``(checks, share after burn-in, λΔt after burn-in)`` of one run."""
    cfg = ControllerConfig(
        tau_s=tau_s,
        rule="A",
        p_star=p_star,
        formula=formula,  # type: ignore[arg-type]
        estimator=estimator,  # type: ignore[arg-type]
        bounds=bounds,
        warmup_s=math.inf,
    )
    trace = simulate(RegimeController(cfg), StepIntensity(rate), horizon_s, seed=seed)
    post = trace.t >= burn_in_tau * tau_s
    share = float(trace.decision[post].mean())
    lam_dt = float(rate * trace.dt[post].mean())
    return len(trace), share, lam_dt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="the small configuration (about 10 s)")
    parser.add_argument("--p-star", type=float, nargs="+", help="targets, in (0, 1)")
    parser.add_argument("--rates", type=float, nargs="+", help="constant true rates, 1/s")
    parser.add_argument("--dev-seeds", type=int, nargs="+", help="development seeds")
    parser.add_argument("--heldout-seeds", type=int, nargs="+", help="held-out seeds")
    parser.add_argument("--horizon", type=float, default=100_000.0, help="run length, s")
    parser.add_argument("--tau", type=float, default=1800.0, help="memory of the estimate, s")
    parser.add_argument("--burn-in-tau", type=float, default=5.0, help="burn-in, in memory times")
    parser.add_argument("--lo", type=float, default=0.01, help="lower interval bound, s")
    parser.add_argument("--hi", type=float, default=1000.0, help="upper interval bound, s")
    parser.add_argument("--csv", type=Path, help="write one row per run to this file")
    return parser


def run_grid(
    p_stars: tuple[float, ...],
    rates: tuple[float, ...],
    dev_seeds: tuple[int, ...],
    heldout_seeds: tuple[int, ...],
    *,
    horizon_s: float,
    tau_s: float,
    burn_in_tau: float,
    bounds: IntervalBounds,
) -> list[RunRow]:
    """Run every combination on the grid and return one row per run."""
    rows: list[RunRow] = []
    seeds = [(s, False) for s in dev_seeds] + [(s, True) for s in heldout_seeds]
    for formula in FORMULAS:
        for estimator in ESTIMATORS:
            for p_star in p_stars:
                for rate in rates:
                    for seed, heldout in seeds:
                        checks, share, lam_dt = run_one(
                            formula,
                            estimator,
                            p_star,
                            rate,
                            seed,
                            horizon_s=horizon_s,
                            tau_s=tau_s,
                            burn_in_tau=burn_in_tau,
                            bounds=bounds,
                        )
                        rows.append(
                            RunRow(
                                formula,
                                estimator,
                                p_star,
                                rate,
                                seed,
                                heldout,
                                checks,
                                share,
                                lam_dt,
                            )
                        )
    return rows


def summarise(rows: list[RunRow]) -> list[dict[str, float | str]]:
    """Aggregate the runs by combination and target."""
    out: list[dict[str, float | str]] = []
    keys = sorted({(r.formula, r.estimator, r.p_star) for r in rows}, key=_order)
    for formula, estimator, p_star in keys:
        key = (formula, estimator, p_star)
        group = [r for r in rows if (r.formula, r.estimator, r.p_star) == key]
        dev = [r.share for r in group if not r.heldout]
        held = [r.share for r in group if r.heldout]
        share_a, lam_dt_a = steady_state_share(p_star, formula, estimator)  # type: ignore[arg-type]
        out.append(
            {
                "formula": formula,
                "estimator": estimator,
                "p_star": p_star,
                "share_analytic": share_a,
                "share_dev": float(np.mean(dev)) if dev else math.nan,
                "share_heldout": float(np.mean(held)) if held else math.nan,
                "max_abs_err": max(abs(r.share - p_star) for r in group),
                "lam_dt_analytic": lam_dt_a,
                "lam_dt_sim": float(np.mean([r.lam_dt for r in group])),
                "checks": float(np.mean([r.checks for r in group])),
            }
        )
    return out


def _order(key: tuple[str, str, float]) -> tuple[int, int, float]:
    return FORMULAS.index(key[0]), ESTIMATORS.index(key[1]), key[2]


def markdown(summary: list[dict[str, float | str]]) -> str:
    """Markdown table of the summary."""
    head = (
        "| formula | estimator | p* | share, analytic | share, dev seeds | "
        "share, held-out | max abs(share − p*) | λΔt, analytic | λΔt, sim | checks/run |"
    )
    lines = [head, "|" + "---|" * 10]
    lines.extend(
        f"| {r['formula']} | {r['estimator']} | {r['p_star']:.2f} | "
        f"{r['share_analytic']:.4f} | {r['share_dev']:.4f} | {r['share_heldout']:.4f} | "
        f"{r['max_abs_err']:.4f} | {r['lam_dt_analytic']:.4f} | {r['lam_dt_sim']:.4f} | "
        f"{r['checks']:.0f} |"
        for r in summary
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[RunRow]) -> None:
    """Write one row per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RunRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        p_stars: tuple[float, ...] = (0.1, 0.3, 0.5)
        rates: tuple[float, ...] = (0.05,)
        dev: tuple[int, ...] = (1, 2)
        held: tuple[int, ...] = (101, 102)
        horizon = 40_000.0
    else:
        p_stars = tuple(args.p_star) if args.p_star else FULL_P_STAR
        rates = tuple(args.rates) if args.rates else FULL_RATES
        dev = tuple(args.dev_seeds) if args.dev_seeds else DEV_SEEDS
        held = tuple(args.heldout_seeds) if args.heldout_seeds else HELDOUT_SEEDS
        horizon = args.horizon
    rows = run_grid(
        p_stars,
        rates,
        dev,
        held,
        horizon_s=horizon,
        tau_s=args.tau,
        burn_in_tau=args.burn_in_tau,
        bounds=IntervalBounds(lo=args.lo, hi=args.hi),
    )
    print(markdown(summarise(rows)))
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
