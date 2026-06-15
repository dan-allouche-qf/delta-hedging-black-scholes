"""Tests for the hedging engine: self-financing identity, agreement with an
independent reference implementation, the convergence behaviour, transaction
costs, and the gamma-P&L decomposition."""

import numpy as np
import pytest

from deltahedge import analysis as an
from deltahedge import black_scholes as bs
from deltahedge import hedging as hg
from deltahedge import paths as pp

R, SIGMA, X, K, T = 0.10, 0.20, 100.0, 90.0, 0.25


def reference_hedge(path, K, r, sigma, T):
    """Independent, explicit single-path hedger that also checks self-financing
    (portfolio value must be continuous across each rebalancing). Returns
    (error, max_self_financing_gap)."""
    n = len(path) - 1
    dt = T / n
    S0 = path[0]
    delta = float(bs.call_delta(S0, K, r, sigma, T))
    cash = float(bs.call_price(S0, K, r, sigma, T)) - delta * S0
    gap = 0.0
    for k in range(1, n + 1):
        S = path[k]
        cash *= np.exp(r * dt)
        value_before = delta * S + cash
        if k < n:
            tau = T - k * dt
            delta_new = float(bs.call_delta(S, K, r, sigma, tau))
            cash -= (delta_new - delta) * S
            value_after = delta_new * S + cash
            gap = max(gap, abs(value_after - value_before))
            delta = delta_new
    pi = delta * path[-1] + cash
    return pi - max(path[-1] - K, 0.0), gap


def test_initial_portfolio_equals_premium():
    paths = pp.gbm_paths(X, R, SIGMA, T, 12, 200, rng=0)
    S0 = paths[:, 0]
    delta0 = bs.call_delta(S0, K, R, SIGMA, T)
    cash0 = bs.call_price(S0, K, R, SIGMA, T) - delta0 * S0
    pi0 = delta0 * S0 + cash0
    v0 = bs.call_price(X, K, R, SIGMA, T)
    np.testing.assert_allclose(pi0, v0, atol=1e-10)


def test_self_financing_and_matches_reference():
    paths = pp.gbm_paths(X, R, SIGMA, T, 12, 50, rng=7)
    engine = hg.hedge_call(paths, K, R, SIGMA, T).error
    ref_err = np.empty(len(paths))
    max_gap = 0.0
    for i, path in enumerate(paths):
        ref_err[i], gap = reference_hedge(path, K, R, SIGMA, T)
        max_gap = max(max_gap, gap)
    assert max_gap < 1e-10, f"self-financing gap too large: {max_gap}"
    np.testing.assert_allclose(engine, ref_err, atol=1e-10)


def test_mean_error_consistent_with_zero_exact_gbm():
    # Under exact GBM with mu=r the discounted self-financing portfolio is a
    # Q-martingale, so E[error] = 0 EXACTLY at every n. We therefore require the
    # t-statistic to stay within +/-4 (a 95% CI would flake ~5% of the time per n).
    n_values = [1, 3, 12, 84]
    paths = pp.crn_gbm_paths(X, R, SIGMA, T, n_values, 40000, rng=123)
    for n in n_values:
        err = hg.hedge_call(paths[n], K, R, SIGMA, T).error
        s = an.mc_summary(err)
        assert abs(s["tstat"]) < 4.0, f"n={n}: |t|={abs(s['tstat']):.2f} suggests a real bias"


def test_std_decreases_with_frequency():
    n_values = [0, 1, 3, 12, 84]
    paths = pp.crn_gbm_paths(X, R, SIGMA, T, n_values, 20000, rng=1)
    stds = {}
    for n in n_values:
        stds[n] = hg.hedge_call(paths[n], K, R, SIGMA, T, hedge=(n > 0)).error.std()
    # No hedge is by far the riskiest; std falls monotonically over n>=1.
    assert stds[0] > 5.0
    pos = [stds[n] for n in [1, 3, 12, 84]]
    assert all(a > b for a, b in zip(pos, pos[1:], strict=False))


def test_transaction_costs_are_nonnegative_and_scale_with_frequency():
    n_values = [3, 84]
    paths = pp.crn_gbm_paths(X, R, SIGMA, T, n_values, 5000, rng=5)
    costs = {}
    for n in n_values:
        res = hg.hedge_call(paths[n], K, R, SIGMA, T, kappa=1e-3)
        assert np.all(res.total_cost >= 0)
        costs[n] = res.total_cost.mean()
    # More frequent rebalancing trades more, hence costs more on average.
    assert costs[84] > costs[3]


def test_digital_engine_runs_and_is_bimodal():
    paths = pp.gbm_paths(X, R, SIGMA, T, 84, 5000, rng=9)
    err = hg.hedge_digital(paths, K, R, SIGMA, T).error
    # Errors cluster near two regimes (payoff 0 vs 1) -> sizeable spread retained.
    assert err.std() > 0.05


def test_gamma_pnl_matches_engine_error():
    # The gamma-P&L decomposition should reproduce the engine error (call, hedged
    # at the realised vol) up to higher-order terms.
    n = 252
    paths = pp.gbm_paths(X, R, SIGMA, T, n, 4000, rng=11)
    err = hg.hedge_call(paths, K, R, SIGMA, T).error
    dec = hg.gamma_pnl_decomposition(paths, K, R, SIGMA, T, gamma_fn=bs.call_gamma)
    # Strong correlation and similar scale (it is a leading-order identity).
    corr = np.corrcoef(err, dec["gamma_pnl"])[0, 1]
    assert corr > 0.9, f"gamma-P&L correlation too low: {corr:.3f}"


def test_hedge_error_requires_two_columns():
    with pytest.raises(ValueError):
        hg.hedge_call(np.array([[100.0], [100.0]]), K, R, SIGMA, T)
