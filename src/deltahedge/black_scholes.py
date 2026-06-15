"""Vectorised Black-Scholes pricing and Greeks for European call and digital
(cash-or-nothing) call options.

All functions accept either scalars or NumPy arrays for the spot ``S`` and are
parameterised by the *time to maturity* ``tau = T - t`` (in years) rather than an
absolute ``(T, t)`` pair, which avoids the ``T``/``t`` confusion present in the
original notebook and is the natural quantity for a hedging engine.

Conventions
-----------
- ``r``      : continuously-compounded risk-free rate.
- ``sigma``  : (annualised) volatility used for *pricing* the option.
- Digital    : cash-or-nothing call paying 1 unit of cash if ``S_T >= K``.
- At ``tau <= 0`` every function returns the option's terminal value / Greek
  limit (intrinsic value for the price, the right-continuous limit otherwise).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "call_price",
    "call_delta",
    "call_gamma",
    "call_vega",
    "call_theta",
    "digital_price",
    "digital_delta",
    "digital_gamma",
    "put_price",
]


def d1_d2(
    S: ArrayLike, K: float, r: float, sigma: float, tau: float
) -> tuple[NDArray, NDArray]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms (vectorised over ``S``)."""
    S = np.asarray(S, dtype=float)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


# --------------------------------------------------------------------------- #
# European call                                                               #
# --------------------------------------------------------------------------- #
def call_price(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Black-Scholes price of a European call."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    d1, d2 = d1_d2(S, K, r, sigma, tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def call_delta(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Delta (dV/dS) of a European call."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return (S > K).astype(float)
    d1, _ = d1_d2(S, K, r, sigma, tau)
    return norm.cdf(d1)


def call_gamma(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Gamma (d2V/dS2) of a European call (identical for call and put)."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.zeros_like(S)
    d1, _ = d1_d2(S, K, r, sigma, tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def call_vega(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Vega (dV/dsigma) of a European call (per unit of volatility)."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.zeros_like(S)
    d1, _ = d1_d2(S, K, r, sigma, tau)
    return S * norm.pdf(d1) * np.sqrt(tau)


def call_theta(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Theta (dV/dt) of a European call (per year). Note dV/dt = -dV/dtau."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.zeros_like(S)
    d1, d2 = d1_d2(S, K, r, sigma, tau)
    term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(tau))
    term2 = -r * K * np.exp(-r * tau) * norm.cdf(d2)
    return term1 + term2


def put_price(S: ArrayLike, K: float, r: float, sigma: float, tau: float) -> NDArray:
    """Black-Scholes price of a European put (used to test put-call parity)."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.maximum(K - S, 0.0)
    d1, d2 = d1_d2(S, K, r, sigma, tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --------------------------------------------------------------------------- #
# Digital (cash-or-nothing) call                                              #
# --------------------------------------------------------------------------- #
def digital_price(
    S: ArrayLike, K: float, r: float, sigma: float, tau: float
) -> NDArray:
    """Price of a cash-or-nothing call paying 1 if ``S_T >= K``."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return (S >= K).astype(float)
    _, d2 = d1_d2(S, K, r, sigma, tau)
    return np.exp(-r * tau) * norm.cdf(d2)


def digital_delta(
    S: ArrayLike, K: float, r: float, sigma: float, tau: float
) -> NDArray:
    """Delta of a digital call. Blows up like 1/sqrt(tau) as ``tau -> 0`` at the
    strike, which is precisely what makes digital options hard to delta-hedge."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.zeros_like(S)
    _, d2 = d1_d2(S, K, r, sigma, tau)
    return np.exp(-r * tau) * norm.pdf(d2) / (S * sigma * np.sqrt(tau))


def digital_gamma(
    S: ArrayLike, K: float, r: float, sigma: float, tau: float
) -> NDArray:
    """Gamma of a digital call. Derived analytically:
    ``-e^{-r tau} phi(d2) d1 / (S^2 sigma^2 tau)``."""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.zeros_like(S)
    d1, d2 = d1_d2(S, K, r, sigma, tau)
    return -np.exp(-r * tau) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * tau)
