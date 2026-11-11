"""Command-line interface: ``trosna-regime-control <command> ...``.

Commands
--------
analyze   summary of a checks.csv trace: rates, model check, controller and
          current heuristic replayed on the same decisions
flicker   exact flicker of the current heuristic at constant rates (Markov chain)

Units: seconds, 1/s for rates, switches per hour. Decimals are dots.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .analyze import TraceReport, analyze
from .controller import ControllerConfig
from .flicker import flicker_stats
from .heuristic import HeuristicConfig, default_config, load_heuristic_config
from .intervals import IntervalBounds
from .trace import read_checks

DEFAULT_RATES = (0.01, 0.02, 0.05, 0.1, 0.2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trosna-regime-control", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse a checks.csv trace")
    a.add_argument("trace", help="path to checks.csv")
    a.add_argument("--tau", type=float, default=1800.0, help="memory of the rate estimate, s")
    a.add_argument("--rule", choices=("A", "B"), default="A", help="interval rule")
    a.add_argument(
        "--p-star",
        type=float,
        default=0.3,
        help="probability of a move in one interval, in (0, 1) (rule A)",
    )
    a.add_argument(
        "--formula",
        choices=("exact", "linear"),
        default="exact",
        help="rule A: exact -ln(1-p*)/rate (default) or linear p*/rate",
    )
    a.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="positive multiplier of the rule A interval (default 1)",
    )
    a.add_argument("--c1-over-c2", type=float, default=2.0, help="C1/C2 in seconds (rule B)")
    a.add_argument("--arl", type=float, default=3600.0, help="detector ARL, s")
    a.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="one-sided confidence of the detector baselines, in [0.5, 1) (default 0.95)",
    )
    a.add_argument(
        "--z",
        type=float,
        help="standard deviations of the baselines; replaces --confidence",
    )
    a.add_argument("--hold", type=float, default=300.0, help="non-stationary hold time, s")
    a.add_argument("--warmup", type=float, default=600.0, help="detector warm-up, s")
    a.add_argument("--heuristic", help="JSON file with the current heuristic table")
    a.add_argument("--burn-in", type=int, default=10, help="checks skipped by the model check")
    a.add_argument("--json", action="store_true", help="print JSON instead of text")

    f = sub.add_parser("flicker", help="exact flicker of the current heuristic")
    f.add_argument("--heuristic", help="JSON file with the current heuristic table")
    f.add_argument(
        "--rate",
        type=float,
        action="append",
        help="intensity 1/s; repeat for several (default: 0.01 … 0.2)",
    )
    f.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser


def _heuristic(path: str | None) -> HeuristicConfig:
    return default_config() if path is None else load_heuristic_config(path)


def _format_report(rep: TraceReport) -> str:
    lines = [
        f"checks                  {rep.n_checks}",
        f"duration                {rep.duration_s:.1f} s",
        f"share of moves          {rep.share_moves:.4f}",
        f"mean interval           {rep.mean_interval_s:.3f} s",
        f"rate, naive N/T         {rep.naive_rate:.5f} 1/s",
        f"rate, debiased          {rep.debiased_rate:.5f} 1/s",
    ]
    if rep.cloglog is None:
        lines.append("model check             not enough checks")
    else:
        fit = rep.cloglog
        lines.append(
            f"slope of ln dt          {fit.slope_ln_dt.estimate:.3f} "
            f"[{fit.slope_ln_dt.ci95[0]:.3f}, {fit.slope_ln_dt.ci95[1]:.3f}] (model: 1)"
        )
        lines.append(
            f"coef of previous bit    {fit.coef_prev_bit.estimate:.3f} "
            f"[{fit.coef_prev_bit.ci95[0]:.3f}, {fit.coef_prev_bit.ci95[1]:.3f}] (model: 0)"
        )
        verdict = "yes" if fit.model_consistent else "no"
        lines.append(f"model consistent        {verdict}")
    lines += [
        f"controller alarms       {rep.controller_alarms}",
        f"controller switches/h   {rep.controller_switches_per_hour:.2f}",
        f"controller nonstat.     {rep.controller_share_nonstationary:.4f}",
        f"current heuristic sw/h  {rep.legacy_switches_per_hour:.2f}",
        f"current heuristic nonst.{rep.legacy_share_nonstationary:.4f}",
    ]
    return "\n".join(lines)


def _cmd_analyze(args: argparse.Namespace) -> int:
    trace = read_checks(args.trace)
    cfg = ControllerConfig(
        tau_s=args.tau,
        rule=args.rule,
        p_star=args.p_star,
        formula=args.formula,
        scale=args.scale,
        c1_over_c2=args.c1_over_c2,
        bounds=IntervalBounds(),
        arl_s=args.arl,
        hold_s=args.hold,
        warmup_s=args.warmup,
        confidence=args.confidence,
        z=args.z,
    )
    rep = analyze(trace, cfg, _heuristic(args.heuristic), burn_in=args.burn_in)
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_report(rep))
    return 0


def _cmd_flicker(args: argparse.Namespace) -> int:
    cfg = _heuristic(args.heuristic)
    if not cfg.is_real:
        print(
            "note: the heuristic table is conditional (is_real = false); results are illustrative",
            file=sys.stderr,
        )
    rates = args.rate or list(DEFAULT_RATES)
    rows = [flicker_stats(r, cfg) for r in rates]
    if args.json:
        print(json.dumps([row.__dict__ for row in rows], indent=2))
        return 0
    print(f"{'rate 1/s':>10} {'switches/h':>11} {'nonstat.':>9} {'interval s':>11} {'checks/h':>9}")
    for row in rows:
        print(
            f"{row.rate:>10.4f} {row.switches_per_hour:>11.2f} "
            f"{row.share_nonstationary:>9.4f} {row.mean_interval_s:>11.3f} "
            f"{row.checks_per_hour:>9.1f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the exit code (2 on a bad input)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return _cmd_analyze(args)
        return _cmd_flicker(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
