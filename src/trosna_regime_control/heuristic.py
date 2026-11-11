"""The 5-bit heuristic of the current system: its rule and its configuration.

After every check the system writes one bit: 1 if memory was moved, 0 if not.
The sum ``S`` of the last five bits sets the interval to the next check (a
hand-made table) and the regime (non-stationary at ``S ≥ threshold``). This
module reproduces the rule for comparison and for migration. It is not
recommended for new deployments: see :doc:`migration <migration>`.

The default table is a conditional one from the document. Real values are
read from a JSON file with :func:`load_heuristic_config`; ``is_real`` in that
file must be ``true`` for real results.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._types import IntArray
from .scheduler import Step

DEFAULT_TABLE: tuple[float, ...] = (10.0, 8.2, 6.4, 4.6, 2.8, 1.0)
DEFAULT_THRESHOLD = 3


@dataclass(frozen=True)
class HeuristicConfig:
    """Table of the heuristic: interval for each sum ``S`` and the regime threshold.

    Attributes
    ----------
    table : tuple of float
        Interval in seconds for ``S = 0 … n_bits``.
    threshold : int
        The regime is non-stationary when ``S ≥ threshold``.
    n_bits : int
        Length of the bit window.
    is_real : bool
        True if the table was taken from the system.

    Examples
    --------
    >>> cfg = HeuristicConfig(DEFAULT_TABLE, DEFAULT_THRESHOLD)
    >>> cfg.interval(5), cfg.mode(2)
    (1.0, False)
    """

    table: tuple[float, ...] = DEFAULT_TABLE
    threshold: int = DEFAULT_THRESHOLD
    n_bits: int = 5
    is_real: bool = False

    def __post_init__(self) -> None:
        """Check the table and the threshold."""
        if len(self.table) != self.n_bits + 1:
            msg = "the table needs n_bits + 1 values (S = 0 … n_bits)"
            raise ValueError(msg)
        if any(dt <= 0.0 for dt in self.table):
            msg = "table intervals must be positive"
            raise ValueError(msg)
        if not 0 <= self.threshold <= self.n_bits + 1:
            msg = "threshold must be between 0 and n_bits + 1"
            raise ValueError(msg)

    def interval(self, s: int) -> float:
        """Interval to the next check for the sum ``s``."""
        return float(self.table[s])

    def mode(self, s: int) -> bool:
        """Return True for the non-stationary regime."""
        return s >= self.threshold


def default_config() -> HeuristicConfig:
    """Return the conditional table of the document (``is_real = False``)."""
    return HeuristicConfig()


def load_heuristic_config(path: str | Path) -> HeuristicConfig:
    """Read a heuristic configuration from a JSON file (UTF-8).

    The file has the keys ``n_bits``, ``interval_by_sum_s`` (a mapping from the
    sum as a string to seconds), ``nonstationary_if_sum_at_least`` and
    ``is_real``. See ``docs/en/cli.md``.
    """
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    n_bits = int(data.get("n_bits", 5))
    table_map: dict[str, float] = data["interval_by_sum_s"]
    table = tuple(float(table_map[str(s)]) for s in range(n_bits + 1))
    return HeuristicConfig(
        table=table,
        threshold=int(data["nonstationary_if_sum_at_least"]),
        n_bits=n_bits,
        is_real=bool(data.get("is_real", False)),
    )


def window_sums(bits: Sequence[int] | IntArray, n_bits: int = 5, initial: int = 0) -> IntArray:
    """Sum ``S`` of the last ``n_bits`` decisions after each check.

    Parameters
    ----------
    bits : sequence of int
        Decisions, 0 or 1, in time order.
    n_bits : int
        Window length.
    initial : int
        Initial window as an integer whose low bit is the latest decision.

    Examples
    --------
    >>> window_sums([1, 1, 0, 0, 0, 0, 1]).tolist()
    [1, 2, 2, 2, 2, 1, 1]
    """
    mask = 2**n_bits - 1
    history = initial & mask
    out = np.empty(len(bits), dtype=np.int64)
    for k, bit in enumerate(bits):
        history = ((history << 1) | int(bit)) & mask
        out[k] = (history).bit_count()
    return out


class LegacyHeuristic:
    """The current 5-bit rule as a scheduler (same interface as the controller).

    Examples
    --------
    >>> legacy = LegacyHeuristic()
    >>> legacy.start(0.0)
    10.0
    >>> legacy.record_decision(10.0, True).next_dt
    8.2
    """

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config if config is not None else default_config()
        self.history = 0

    @property
    def s_sum(self) -> int:
        """Current sum ``S``."""
        return self.history.bit_count()

    def start(self, t0: float) -> float:
        """Return the interval for the empty window."""
        del t0
        return self.config.interval(self.s_sum)

    def record_decision(self, t: float, decision: bool) -> Step:
        """Shift the decision into the window and return the table interval."""
        mask = 2**self.config.n_bits - 1
        self.history = ((self.history << 1) | int(decision)) & mask
        s = self.s_sum
        dt = self.config.interval(s)
        return Step(
            t_next=t + dt,
            next_dt=dt,
            nonstationary=self.config.mode(s),
            alarm=None,
            rate=math.nan,
        )
