"""Statistical analysis helpers: Monte-Carlo summaries with **standard errors**
and confidence intervals, a proper power-law fit of the ``std ~ n^b`` scaling
law, and tail-risk metrics. These exist specifically to stop the original
project's habit of quoting Monte-Carlo noise to 4-6 decimals as if it were a
resolved effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "mc_summary",
    "summary_table",
    "fit_power_law",
    "var_cvar",
]


def mc_summary(errors: NDArray, z: float = 1.959963984540054) -> dict[str, float]:
    """Monte-Carlo summary of a sample of hedging errors.

    Reports the mean, its standard error ``SE = std / sqrt(N)``, a ``z``-based
    confidence interval for the mean, the t-statistic and two-sided p-value of
    the test ``mean == 0``, plus std/min/max/percentiles. ``mean_is_zero`` flags
    whether 0 lies inside the CI (i.e. the mean is indistinguishable from zero).
    """
    errors = np.asarray(errors, dtype=float)
    n = errors.size
    mean = float(errors.mean())
    std = float(errors.std(ddof=1))
    sem = std / np.sqrt(n)
    tstat = mean / sem if sem > 0 else 0.0
    pvalue = float(2 * stats.t.sf(abs(tstat), df=n - 1)) if sem > 0 else 1.0
    return {
        "n_paths": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - z * sem,
        "ci_high": mean + z * sem,
        "tstat": tstat,
        "pvalue": pvalue,
        "mean_is_zero": bool(mean - z * sem <= 0 <= mean + z * sem),
        "min": float(errors.min()),
        "max": float(errors.max()),
        "p5": float(np.percentile(errors, 5)),
        "p95": float(np.percentile(errors, 95)),
    }


def summary_table(results: dict[int, NDArray]) -> pd.DataFrame:
    """Build a tidy summary table across rebalancing frequencies ``n``.

    ``results`` maps ``n -> errors``. The returned DataFrame is indexed by ``n``
    and carries the mean with its standard error, the CI, std, p-value, and the
    ``mean_is_zero`` flag, so every reported figure is shown with its precision.
    """
    rows = []
    for n in sorted(results):
        s = mc_summary(results[n])
        rows.append(
            {
                "n": n,
                "mean": s["mean"],
                "sem": s["sem"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "std": s["std"],
                "pvalue": s["pvalue"],
                "mean_is_zero": s["mean_is_zero"],
                "min": s["min"],
                "max": s["max"],
            }
        )
    return pd.DataFrame(rows).set_index("n")


def fit_power_law(
    n_values: list[int], stds: list[float], exclude: tuple[int, ...] = (1,)
) -> dict[str, float]:
    """Fit ``log(std) = a + b * log(n)`` by OLS and return the exponent ``b``.

    ``n=1`` is excluded by default: it performs no intermediate rebalancing (a
    static hedge held to maturity) and lies outside the asymptotic ``1/sqrt(n)``
    regime. For a vanilla call the fitted ``b`` should be ~ -0.5; a value far
    from -0.5 (as for digitals, ~ -0.23) is a genuine qualitative finding.

    Returns ``slope``, its standard error and 95% CI, and ``intercept``.
    """
    pairs = [(n, s) for n, s in zip(n_values, stds, strict=True) if n not in exclude]
    if len(pairs) < 2:
        raise ValueError("Need at least two points (after exclusion) to fit.")
    logn = np.log([p[0] for p in pairs])
    logs = np.log([p[1] for p in pairs])
    # OLS with covariance for the slope's standard error.
    coeffs, cov = np.polyfit(logn, logs, deg=1, cov=True)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    slope_se = float(np.sqrt(cov[0, 0]))
    return {
        "slope": slope,
        "slope_se": slope_se,
        "slope_ci_low": slope - 1.96 * slope_se,
        "slope_ci_high": slope + 1.96 * slope_se,
        "intercept": intercept,
        "n_used": [p[0] for p in pairs],
    }


def var_cvar(pnl: NDArray, alpha: float = 0.05) -> dict[str, float]:
    """Value-at-Risk and Conditional VaR (Expected Shortfall) of the *loss*
    distribution ``loss = -pnl`` at level ``alpha`` (e.g. 5%).

    ``pnl`` is the hedger's terminal P&L (the engine's ``error``). VaR is the
    ``1-alpha`` loss quantile; CVaR is the mean loss beyond it.
    """
    pnl = np.asarray(pnl, dtype=float)
    loss = -pnl
    var = float(np.quantile(loss, 1 - alpha))
    tail = loss[loss >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"alpha": alpha, "var": var, "cvar": cvar}
