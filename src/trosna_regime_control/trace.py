"""Traces of checks: the log format of the system and its validation.

One row per check, columns (see ``docs/en/log_format.md``):

============ ===================================================================
``t_unix``    moment of the check, seconds since the Unix epoch (required)
``b``         decision, 0 or 1 (required)
``dt_s``      real interval before the check, seconds (optional; computed from
              ``t_unix`` if absent, and then the first row is dropped)
``s_sum``     sum of the 5-bit heuristic after the check (optional)
``mode``      regime of the current heuristic, 0 or 1 (optional)
============ ===================================================================

Other columns of the log (``next_dt_s``, ``cycle_cost_s``, ``moved_mb``) are
ignored here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._types import FloatArray, IntArray


@dataclass(frozen=True)
class Trace:
    """A sequence of checks, times in seconds from the start of observation.

    Attributes
    ----------
    t : numpy.ndarray
        Check times, strictly increasing.
    dt : numpy.ndarray
        Real interval before each check, positive.
    decision : numpy.ndarray
        Decision bits, 0 or 1.
    legacy_mode : numpy.ndarray or None
        Regime of the current heuristic as logged, if present.
    s_sum : numpy.ndarray or None
        Sum of the current heuristic as logged, if present.
    """

    t: FloatArray
    dt: FloatArray
    decision: IntArray
    legacy_mode: IntArray | None = None
    s_sum: IntArray | None = None

    def __post_init__(self) -> None:
        """Validate the arrays."""
        n = self.t.shape[0]
        if self.dt.shape != (n,) or self.decision.shape != (n,):
            msg = "t, dt and decision must have the same length"
            raise ValueError(msg)
        if n == 0:
            msg = "the trace is empty"
            raise ValueError(msg)
        if np.any(np.diff(self.t) <= 0.0):
            msg = "check times must be strictly increasing"
            raise ValueError(msg)
        if not np.all(np.isfinite(self.dt)) or np.any(self.dt <= 0.0):
            msg = "intervals must be finite and positive"
            raise ValueError(msg)
        if not np.all(np.isin(self.decision, (0, 1))):
            msg = "decisions must be 0 or 1"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of checks."""
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        """Time from the start of observation to the last check, seconds."""
        return float(self.t[-1])


def read_checks(path: str | Path) -> Trace:
    r"""Read a ``checks.csv`` file of the system (UTF-8) into a :class:`Trace`.

    Raises
    ------
    ValueError
        If the file is empty, a required column is missing, the times are not
        increasing, an interval is not positive, or a decision is not 0 or 1.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (d / "c.csv").write_text("t_unix,dt_s,b\n100,10,0\n101,1,1\n", encoding="utf-8")
    >>> read_checks(d / "c.csv").decision.tolist()
    [0, 1]
    """
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        msg = f"empty file: {path}"
        raise ValueError(msg)
    if "t_unix" not in rows[0] or "b" not in rows[0]:
        msg = "the columns t_unix and b are required"
        raise ValueError(msg)
    t_unix = np.array([float(r["t_unix"]) for r in rows], dtype=np.float64)
    decision = np.array([int(float(r["b"])) for r in rows], dtype=np.int64)
    if "dt_s" in rows[0] and rows[0]["dt_s"] != "":
        dt = np.array([float(r["dt_s"]) for r in rows], dtype=np.float64)
        start = float(t_unix[0] - dt[0])
        keep = slice(None)
    else:
        if len(rows) < 2:
            msg = "without dt_s at least two rows are needed"
            raise ValueError(msg)
        dt = np.diff(t_unix)
        start = float(t_unix[0])
        keep = slice(1, None)
        t_unix = t_unix[1:]
        decision = decision[1:]
    legacy = _optional_int(rows, "mode", keep)
    sums = _optional_int(rows, "s_sum", keep)
    return Trace(
        t=np.asarray(t_unix - start, dtype=np.float64),
        dt=dt,
        decision=decision,
        legacy_mode=legacy,
        s_sum=sums,
    )


def _optional_int(rows: list[dict[str, str]], name: str, keep: slice) -> IntArray | None:
    if name not in rows[0] or rows[0][name] == "":
        return None
    return np.array([int(float(r[name])) for r in rows], dtype=np.int64)[keep]
