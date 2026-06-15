"""Path simulators for the underlying asset.

The default model is geometric Brownian motion simulated with the **exact
log-Euler** scheme

    S_{k+1} = S_k * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),

which is exact for GBM (no discretisation bias and no risk of negative prices),
unlike the arithmetic Euler-Maruyama scheme used for ``n>0`` in the original
notebook. Heston (stochastic volatility) and Merton (jump-diffusion) simulators
are provided to study model-misspecification risk.

All randomness goes through an explicit ``numpy.random.Generator`` so results are
reproducible regardless of call order (no reliance on a single global seed).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

__all__ = [
    "gbm_paths",
    "gbm_paths_from_increments",
    "crn_gbm_paths",
    "heston_paths",
    "merton_paths",
]


def _as_rng(rng: Generator | int | None) -> Generator:
    if isinstance(rng, Generator):
        return rng
    return default_rng(rng)


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: Generator | int | None = None,
) -> NDArray:
    """Simulate GBM paths with the exact log-Euler scheme.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first column is
    ``S0``. ``n_steps == 0`` returns a single terminal column drawn directly.
    """
    rng = _as_rng(rng)
    if n_steps == 0:
        Z = rng.standard_normal(n_paths)
        S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return np.column_stack([np.full(n_paths, S0), S_T])
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    return gbm_paths_from_increments(S0, mu, sigma, dt, Z)


def gbm_paths_from_increments(
    S0: float, mu: float, sigma: float, dt: float, Z: NDArray
) -> NDArray:
    """Build exact GBM paths from a pre-drawn ``(n_paths, n_steps)`` array of
    standard normals ``Z``. Used by the common-random-numbers helper."""
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_path = np.cumsum(log_increments, axis=1)
    S = S0 * np.exp(log_path)
    return np.column_stack([np.full(Z.shape[0], S0), S])


def _lcm(values: list[int]) -> int:
    out = 1
    for v in values:
        out = out * v // gcd(out, v)
    return out


def crn_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_values: list[int],
    n_paths: int,
    rng: Generator | int | None = None,
) -> dict[int, NDArray]:
    """Simulate GBM paths for several rebalancing frequencies driven by the
    **same** underlying Brownian motion (common random numbers).

    A fine grid (the lcm of all ``n>0`` values) is drawn once; coarser paths are
    obtained by aggregating the fine Gaussian increments in blocks. Because the
    aggregation is exact for GBM, every returned path is a genuine exact-GBM path
    and all frequencies see identical terminal prices for the same scenario,
    which removes Monte-Carlo noise from cross-frequency comparisons.

    Returns ``{n: path}`` with ``path`` of shape ``(n_paths, n + 1)``.
    """
    rng = _as_rng(rng)
    positive = sorted({n for n in n_values if n > 0})
    out: dict[int, NDArray] = {}

    if not positive:
        n_fine = 1
        Z_fine = np.empty((n_paths, 0))
    else:
        n_fine = _lcm(positive)
        Z_fine = rng.standard_normal((n_paths, n_fine))

    dt_fine = T / n_fine
    for n in n_values:
        if n == 0:
            # Terminal price taken from the fine path so n=0 shares the scenario.
            if n_fine == 0 or Z_fine.shape[1] == 0:
                Z = rng.standard_normal((n_paths, 1))
                out[0] = gbm_paths_from_increments(S0, mu, sigma, T, Z)[:, [0, -1]]
            else:
                fine = gbm_paths_from_increments(S0, mu, sigma, dt_fine, Z_fine)
                out[0] = fine[:, [0, -1]]
            continue
        block = n_fine // n
        # Aggregate `block` fine standard normals into one coarse standard normal.
        Z_coarse = Z_fine.reshape(n_paths, n, block).sum(axis=2) / np.sqrt(block)
        dt = T / n
        out[n] = gbm_paths_from_increments(S0, mu, sigma, dt, Z_coarse)
    return out


def heston_paths(
    S0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: Generator | int | None = None,
) -> NDArray:
    """Simulate Heston stochastic-volatility price paths (full-truncation Euler
    on the variance). Returns shape ``(n_paths, n_steps + 1)``."""
    rng = _as_rng(rng)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for k in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW1 = sqrt_dt * z1
        dW2 = sqrt_dt * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
        v_pos = np.maximum(v[:, k], 0.0)
        v[:, k + 1] = (
            v[:, k]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * dW2
        )
        S[:, k + 1] = S[:, k] * np.exp(
            (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW1
        )
    return S


def merton_paths(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    jump_mean: float,
    jump_std: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: Generator | int | None = None,
) -> NDArray:
    """Simulate Merton jump-diffusion paths. Jumps are compound-Poisson with
    log-normal sizes; the drift is compensated so ``E[S_t] = S0 e^{mu t}``.
    Returns shape ``(n_paths, n_steps + 1)``."""
    rng = _as_rng(rng)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    k_bar = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0  # E[e^J - 1]
    drift = (mu - 0.5 * sigma**2 - lam * k_bar) * dt
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, n_paths)
        # Sum of n_jumps i.i.d. N(jump_mean, jump_std^2) jump sizes.
        jump = rng.normal(
            n_jumps * jump_mean, np.sqrt(n_jumps) * jump_std
        )
        log_inc = drift + sigma * sqrt_dt * z + jump
        S[:, k + 1] = S[:, k] * np.exp(log_inc)
    return S
