"""Check of the decision model ``P(b_k = 1 | past) = 1 − e^{−λΔt_k}`` (experiment Э1).

On the complementary log-log scale the model is linear:
``ln(−ln(1 − p)) = ln λ + ln Δt``. Regress ``b_k`` on ``1``, ``ln Δt_k`` and
``b_{k−1}`` with the cloglog link. If the model holds, the slope of ``ln Δt``
is 1, the coefficient of the previous bit is 0, and the intercept is ``ln λ``.

The fit is a Fisher-scoring (IRLS) iteration written with numpy only; no
statsmodels is needed. Standard errors come from the inverse information
matrix, so the intervals are asymptotic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import FloatArray, IntArray


@dataclass(frozen=True)
class Coefficient:
    """Estimate and asymptotic 95 % interval of a regression coefficient."""

    estimate: float
    ci95: tuple[float, float]

    def contains(self, value: float) -> bool:
        """Return True if ``value`` lies in the 95 % interval."""
        return self.ci95[0] <= value <= self.ci95[1]


@dataclass(frozen=True)
class CloglogFit:
    """Result of :func:`fit_cloglog`.

    Attributes
    ----------
    intercept : Coefficient
        ``ln λ`` (at ``Δt = 1 s``).
    slope_ln_dt : Coefficient
        Slope of ``ln Δt``; 1 under the model.
    coef_prev_bit : Coefficient
        Effect of ``b_{k−1}``; 0 under the model (no carry-over).
    rate_hat : float
        Intensity under the model with the slope fixed at 1, 1/s.
    n_obs : int
        Number of checks used.
    converged : bool
        Whether the iteration converged.
    """

    intercept: Coefficient
    slope_ln_dt: Coefficient
    coef_prev_bit: Coefficient
    rate_hat: float
    n_obs: int
    converged: bool

    @property
    def model_consistent(self) -> bool:
        """True if the slope interval contains 1 and the previous-bit interval contains 0."""
        return self.slope_ln_dt.contains(1.0) and self.coef_prev_bit.contains(0.0)


def _irls(
    y: FloatArray, x: FloatArray, offset: FloatArray, n_iter: int = 100, tol: float = 1e-10
) -> tuple[FloatArray, FloatArray, bool]:
    """Fisher scoring for the binomial model with the cloglog link and a known offset.

    The linear predictor is ``η = x β + offset``. Returns the coefficients, the
    covariance matrix of the last step and the convergence flag.
    """
    eps = 1e-12
    mu0 = np.clip((y + 0.5) / 2.0, eps, 1.0 - eps)
    eta0 = np.log(-np.log1p(-mu0))
    beta = np.linalg.lstsq(x, eta0 - offset, rcond=None)[0]
    cov = np.eye(x.shape[1])
    for _ in range(n_iter):
        eta = x @ beta + offset
        e = np.exp(np.clip(eta, -50.0, 50.0))
        mu = np.clip(-np.expm1(-e), eps, 1.0 - eps)
        dmu = np.maximum(e * np.exp(-e), eps)
        w = dmu**2 / (mu * (1.0 - mu))
        z = (eta - offset) + (y - mu) / dmu
        info = x.T @ (w[:, None] * x)
        new = np.linalg.solve(info, x.T @ (w * z))
        cov = np.linalg.inv(info)
        step = float(np.max(np.abs(new - beta)))
        beta = new
        if step < tol:
            return beta, cov, True
    return beta, cov, False


def _coefficient(beta: FloatArray, se: FloatArray, i: int) -> Coefficient:
    half = 1.959964 * float(se[i])
    return Coefficient(estimate=float(beta[i]), ci95=(float(beta[i]) - half, float(beta[i]) + half))


def fit_cloglog(decisions: npt.ArrayLike, dt: npt.ArrayLike, burn_in: int = 0) -> CloglogFit:
    """Fit the decision model with the cloglog link.

    Two fits are made on the same checks. The full model
    ``b_k ~ 1 + ln Δt_k + b_{k−1}`` gives the coefficients and the model check.
    The restricted model with the slope fixed at 1 (``ln Δt_k`` as an offset)
    gives ``rate_hat = exp(intercept)``, the intensity under the model. It has a
    much smaller variance than the intercept of the full fit, which is
    extrapolated to ``Δt = 1 s``.

    Parameters
    ----------
    decisions : array_like
        Decision bits, 0 or 1, in time order.
    dt : array_like
        Real intervals before each check, seconds.
    burn_in : int
        Number of initial checks to skip.

    Raises
    ------
    ValueError
        If the inputs differ in length or fewer than four usable checks are left.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> dts = rng.uniform(1.0, 10.0, 4000)
    >>> bits = (rng.random(4000) < -np.expm1(-0.05 * dts)).astype(int)
    >>> fit = fit_cloglog(bits, dts, burn_in=1)
    >>> fit.model_consistent
    True
    """
    b = np.asarray(decisions, dtype=np.float64)
    d = np.asarray(dt, dtype=np.float64)
    if b.shape != d.shape:
        msg = "decisions and dt must have the same length"
        raise ValueError(msg)
    if b.shape[0] - burn_in - 1 < 4:
        msg = "too few checks for the fit"
        raise ValueError(msg)
    y = b[burn_in + 1 :]
    log_dt = np.log(d[burn_in + 1 :])
    prev = b[burn_in:-1]
    full_x = np.column_stack([np.ones_like(y), log_dt, prev])
    beta, cov, converged = _irls(y, full_x, np.zeros_like(y))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    fixed_x = np.column_stack([np.ones_like(y), prev])
    beta_fixed, _, converged_fixed = _irls(y, fixed_x, log_dt)
    return CloglogFit(
        intercept=_coefficient(beta, se, 0),
        slope_ln_dt=_coefficient(beta, se, 1),
        coef_prev_bit=_coefficient(beta, se, 2),
        rate_hat=float(math.exp(beta_fixed[0])),
        n_obs=int(y.shape[0]),
        converged=bool(converged and converged_fixed),
    )


@dataclass(frozen=True)
class IntervalShares:
    """Share of moves by groups of intervals, for the comparison with ``1 − e^{−λΔt}``.

    Attributes
    ----------
    dt : numpy.ndarray
        Mean interval of each group, seconds.
    share : numpy.ndarray
        Share of checks with a move in each group.
    count : numpy.ndarray
        Number of checks in each group.
    """

    dt: FloatArray
    share: FloatArray
    count: IntArray


def ones_share_by_interval(
    decisions: npt.ArrayLike, dt: npt.ArrayLike, n_bins: int = 10
) -> IntervalShares:
    """Share of moves by quantile groups of the interval.

    If there are at most 20 distinct intervals, each value is a group.

    Examples
    --------
    >>> out = ones_share_by_interval([0, 1, 1, 0], [1.0, 1.0, 2.0, 2.0])
    >>> out.share.tolist()
    [0.5, 0.5]
    """
    b = np.asarray(decisions, dtype=np.float64)
    d = np.asarray(dt, dtype=np.float64)
    values = np.unique(d)
    if values.size <= 20:
        groups = [d == v for v in values]
        centers = values
    else:
        edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
        idx = np.clip(np.digitize(d, edges[1:-1]), 0, n_bins - 1)
        groups = [idx == i for i in range(n_bins)]
        centers = np.array([d[g].mean() for g in groups])
    share = np.array([b[g].mean() if g.any() else math.nan for g in groups])
    count = np.array([int(g.sum()) for g in groups], dtype=np.int64)
    return IntervalShares(
        dt=np.asarray(centers, dtype=np.float64),
        share=np.asarray(share, dtype=np.float64),
        count=count,
    )
