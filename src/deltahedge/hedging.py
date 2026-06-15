"""Vectorised discrete-time delta-hedging engine.

The engine replicates a **short** option position with a self-financing
portfolio of the underlying and a money-market account, rebalanced at the start
of each of ``n`` intervals (and held to maturity over the last one — there is no
useless trade at expiry). It is parameterised by the option's ``price`` and
``delta`` functions so the *same* engine handles vanilla calls, digitals, or any
other European payoff.

The reported quantity is the hedger's terminal P&L

    error = pi(T) - payoff,

which is ~0 in the continuous-hedging limit. Optional proportional transaction
costs and a hedging volatility distinct from the realised one let us study the
cost/variance trade-off and volatility-misspecification risk.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import black_scholes as bs

__all__ = [
    "HedgeResult",
    "hedge_error",
    "hedge_call",
    "hedge_digital",
    "call_payoff",
    "digital_payoff",
    "gamma_pnl_decomposition",
    "leland_vol",
    "pnl_attribution",
]


def call_payoff(S_T: NDArray, K: float) -> NDArray:
    return np.maximum(S_T - K, 0.0)


def digital_payoff(S_T: NDArray, K: float) -> NDArray:
    return (S_T >= K).astype(float)


class HedgeResult:
    """Container for the output of :func:`hedge_error`."""

    __slots__ = ("error", "total_cost", "v0", "n_steps")

    def __init__(
        self, error: NDArray, total_cost: NDArray, v0: float, n_steps: int
    ) -> None:
        self.error = error
        self.total_cost = total_cost
        self.v0 = v0
        self.n_steps = n_steps

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HedgeResult(n_steps={self.n_steps}, V0={self.v0:.4f}, "
            f"mean_error={self.error.mean():.4f}, std_error={self.error.std():.4f}, "
            f"mean_cost={self.total_cost.mean():.4f})"
        )


def hedge_error(
    paths: NDArray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    price_fn: Callable[..., NDArray],
    delta_fn: Callable[..., NDArray],
    payoff_fn: Callable[[NDArray, float], NDArray],
    kappa: float = 0.0,
    charge_setup_cost: bool = True,
    hedge: bool = True,
) -> HedgeResult:
    """Run the discrete delta hedge over pre-simulated ``paths``.

    Parameters
    ----------
    paths : array ``(n_paths, n_steps + 1)``
        Realised price scenarios (column 0 is ``S0``). The number of rebalancing
        intervals is ``n_steps = n_cols - 1``; the delta is set at the start of
        each interval and the position is held over the last one (no useless
        trade at expiry).
    sigma_hedge : float
        Volatility used to compute the option price and delta (the hedger's
        *implied* vol). Set it different from the vol that generated ``paths`` to
        study volatility misspecification.
    kappa : float
        Proportional transaction-cost rate charged on traded notional.
    charge_setup_cost : bool
        Whether the initial hedge set-up is also charged ``kappa``.
    hedge : bool
        If ``False``, hold zero shares throughout (the "no hedging" benchmark:
        sell the option and invest the premium at ``r``). Pass a 2-column path
        ``[S0, S_T]`` for this case.

    Notes
    -----
    The drift used to generate ``paths`` never enters the hedge: the engine only
    sees realised prices, so the same realised scenario gives the same error
    whatever the drift that produced it.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    if n_steps < 1:
        raise ValueError("paths must have at least 2 columns ([S0, S_T]).")
    S0 = paths[:, 0]
    dt = T / n_steps

    v0 = float(price_fn(S0[0], K, r, sigma_hedge, T))
    total_cost = np.zeros(n_paths)

    if not hedge:
        # No hedging: hold zero shares, premium compounds at the risk-free rate.
        pi = v0 * np.exp(r * T)
        S_T = paths[:, -1]
        error = pi - payoff_fn(S_T, K)
        return HedgeResult(error, total_cost, v0, 0)

    delta = price_and_delta_init(delta_fn, S0, K, r, sigma_hedge, T)
    cash = v0 - delta * S0
    if kappa and charge_setup_cost:
        setup = kappa * np.abs(delta) * S0
        cash -= setup
        total_cost += setup

    for k in range(1, n_steps + 1):
        S = paths[:, k]
        cash = cash * np.exp(r * dt)
        if k < n_steps:
            tau = T - k * dt
            delta_new = delta_fn(S, K, r, sigma_hedge, tau)
            trade = delta_new - delta
            if kappa:
                cost = kappa * np.abs(trade) * S
                cash -= cost
                total_cost += cost
            cash -= trade * S
            delta = delta_new

    S_T = paths[:, -1]
    pi = delta * S_T + cash
    error = pi - payoff_fn(S_T, K)
    return HedgeResult(error, total_cost, v0, n_steps)


def price_and_delta_init(delta_fn, S0, K, r, sigma_hedge, T) -> NDArray:
    """Initial delta as an array (kept as a helper for clarity/testability)."""
    return np.asarray(delta_fn(S0, K, r, sigma_hedge, T), dtype=float)


def hedge_call(
    paths: NDArray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    kappa: float = 0.0,
    hedge: bool = True,
) -> HedgeResult:
    """Delta-hedge a European call over ``paths`` (``hedge=False`` = no hedging)."""
    return hedge_error(
        paths, K, r, sigma_hedge, T,
        bs.call_price, bs.call_delta, call_payoff, kappa=kappa, hedge=hedge,
    )


def hedge_digital(
    paths: NDArray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    kappa: float = 0.0,
    hedge: bool = True,
) -> HedgeResult:
    """Delta-hedge a digital (cash-or-nothing) call (``hedge=False`` = no hedging)."""
    return hedge_error(
        paths, K, r, sigma_hedge, T,
        bs.digital_price, bs.digital_delta, digital_payoff, kappa=kappa, hedge=hedge,
    )


def gamma_pnl_decomposition(
    paths: NDArray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    gamma_fn: Callable[..., NDArray] = bs.call_gamma,
) -> dict[str, NDArray]:
    """Decompose the discrete hedging P&L into its gamma-P&L components.

    For a short option delta-hedged at ``sigma_hedge``, the terminal hedging
    error is, to leading order,

        error ~ -0.5 * sum_k Gamma_k * S_k^2 * [ (dS_k / S_k)^2 - sigma_hedge^2 * dt ]

    which splits into a **deterministic variance-mismatch** term
    (``-0.5 Gamma_k S_k^2 (sigma_real^2 - sigma_hedge^2) dt`` in expectation) and
    a **mean-zero discretisation noise** term ``~ (Z_k^2 - 1)``. This is the
    classical "robustness of Black-Scholes" identity and explains why hedge-P&L
    is driven by realised-vs-implied variance rather than drift.

    Returns per-path arrays: ``gamma_pnl`` (the full sum, comparable to the
    engine's ``error``), ``mismatch`` (deterministic term using the realised
    squared returns) and ``noise`` (the remainder).
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    S = paths[:, :-1]                       # S_k at k = 0 .. n-1
    dS = np.diff(paths, axis=1)             # S_{k+1} - S_k
    taus = T - np.arange(n_steps) * dt      # time to maturity at each step
    gammas = np.stack(
        [gamma_fn(S[:, k], K, r, sigma_hedge, taus[k]) for k in range(n_steps)],
        axis=1,
    )
    realised_var = (dS / S) ** 2
    bracket = realised_var - sigma_hedge**2 * dt
    weight = 0.5 * gammas * S**2
    gamma_pnl = -np.sum(weight * bracket, axis=1)
    # Deterministic part replaces realised squared return by its dt-expectation.
    mismatch = -np.sum(weight * (realised_var.mean(axis=0) - sigma_hedge**2 * dt), axis=1)
    noise = gamma_pnl - mismatch
    return {"gamma_pnl": gamma_pnl, "mismatch": mismatch, "noise": noise}


def leland_vol(sigma: float, kappa: float, dt: float, sign: int = 1) -> float:
    """Leland (1985) transaction-cost-adjusted hedging volatility.

    ``sigma_modified = sigma * sqrt(1 + sign * A)`` with the Leland number
    ``A = sqrt(2/pi) * kappa / (sigma * sqrt(dt))``. A long-gamma hedger inflates
    vol (``sign=+1``); a short-gamma writer deflates it (``sign=-1``). Hedging at
    the modified vol trades less aggressively where it is expensive and improves
    the cost-adjusted P&L distribution.
    """
    A = np.sqrt(2.0 / np.pi) * kappa / (sigma * np.sqrt(dt))
    inside = 1.0 + sign * A
    if inside <= 0:
        raise ValueError("Leland adjustment drives variance non-positive; reduce kappa.")
    return float(sigma * np.sqrt(inside))


def pnl_attribution(
    paths: NDArray, K: float, r: float, sigma: float, T: float
) -> dict[str, NDArray]:
    """Greeks-based P&L attribution of a long call's value change per step:

        dV ~ theta*dt + delta*dS + 0.5*gamma*dS^2  (+ residual).

    Returns the path-summed ``theta``, ``delta`` and ``gamma`` contributions, the
    ``predicted`` total, the ``actual`` value change ``V(T)-V(0)`` and the
    ``residual``. A small residual relative to the gamma term confirms the
    second-order Taylor attribution that desks use to explain daily P&L.
    """
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    S = paths[:, :-1]
    dS = np.diff(paths, axis=1)
    taus = T - np.arange(n_steps) * dt
    theta = np.stack([bs.call_theta(S[:, k], K, r, sigma, taus[k]) for k in range(n_steps)], axis=1)
    delta = np.stack([bs.call_delta(S[:, k], K, r, sigma, taus[k]) for k in range(n_steps)], axis=1)
    gamma = np.stack([bs.call_gamma(S[:, k], K, r, sigma, taus[k]) for k in range(n_steps)], axis=1)

    theta_c = np.sum(theta * dt, axis=1)
    delta_c = np.sum(delta * dS, axis=1)
    gamma_c = np.sum(0.5 * gamma * dS**2, axis=1)
    predicted = theta_c + delta_c + gamma_c
    v0 = bs.call_price(paths[:, 0], K, r, sigma, T)
    vT = np.maximum(paths[:, -1] - K, 0.0)
    actual = vT - v0
    return {
        "theta": theta_c,
        "delta": delta_c,
        "gamma": gamma_c,
        "predicted": predicted,
        "actual": actual,
        "residual": actual - predicted,
    }
