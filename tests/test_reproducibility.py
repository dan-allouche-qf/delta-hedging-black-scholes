"""Reproducibility tests. The whole point of moving to ``np.random.default_rng``
with per-call seeding is that results no longer depend on global state or the
order in which simulations are run."""

import numpy as np

from deltahedge import hedging as hg
from deltahedge import paths as pp

R, SIGMA, X, K, T = 0.10, 0.20, 100.0, 90.0, 0.25


def test_same_seed_same_paths():
    a = pp.gbm_paths(X, R, SIGMA, T, 12, 1000, rng=2024)
    b = pp.gbm_paths(X, R, SIGMA, T, 12, 1000, rng=2024)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_paths():
    a = pp.gbm_paths(X, R, SIGMA, T, 12, 1000, rng=1)
    b = pp.gbm_paths(X, R, SIGMA, T, 12, 1000, rng=2)
    assert not np.array_equal(a, b)


def test_order_independence():
    # Running an unrelated simulation in between must not change a seeded result,
    # unlike the original global-seed setup where cell order changed every figure.
    first = pp.gbm_paths(X, R, SIGMA, T, 84, 500, rng=99)
    _ = pp.gbm_paths(X, 0.3, 0.5, T, 10, 7777, rng=None)  # noise with its own rng
    again = pp.gbm_paths(X, R, SIGMA, T, 84, 500, rng=99)
    np.testing.assert_array_equal(first, again)


def test_crn_shares_terminal_across_frequencies():
    # Common random numbers: every frequency must see the same terminal prices
    # for the same scenario (exact-GBM aggregation).
    n_values = [0, 1, 3, 12, 84]
    paths = pp.crn_gbm_paths(X, R, SIGMA, T, n_values, 1000, rng=42)
    terminal = {n: paths[n][:, -1] for n in n_values}
    ref = terminal[84]
    for n in n_values:
        np.testing.assert_allclose(terminal[n], ref, atol=1e-9)


def test_hedge_result_reproducible():
    p1 = pp.crn_gbm_paths(X, R, SIGMA, T, [12], 2000, rng=55)[12]
    p2 = pp.crn_gbm_paths(X, R, SIGMA, T, [12], 2000, rng=55)[12]
    e1 = hg.hedge_call(p1, K, R, SIGMA, T).error
    e2 = hg.hedge_call(p2, K, R, SIGMA, T).error
    np.testing.assert_array_equal(e1, e2)
