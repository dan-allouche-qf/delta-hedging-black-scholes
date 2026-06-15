"""Regression tests pinning the project's headline numbers so refactors cannot
silently change the results. Values are produced with a fixed seed and the exact
GBM scheme (NumPy's PCG64 ``default_rng`` is stable across versions)."""

import numpy as np

import deltahedge as dh
from deltahedge import analysis as an
from deltahedge import hedging as hg

R, SIGMA, X, K, T = 0.10, 0.20, 100.0, 90.0, 0.25
N = 20_000
SEED = 20_240_614

# Pinned standard deviations of the hedging error (exact GBM, mu=r, CRN across n).
CALL_STD = {0: 9.573426, 1: 1.438637, 3: 0.963658, 12: 0.520341, 84: 0.205257}
DIGITAL_STD = {
    0: 0.303640, 1: 0.254419, 3: 0.197231, 12: 0.143402, 84: 0.089374, 252: 0.066353,
}


def test_call_std_regression():
    paths = dh.crn_gbm_paths(X, R, SIGMA, T, list(CALL_STD), N, rng=SEED)
    for n, expected in CALL_STD.items():
        std = hg.hedge_call(paths[n], K, R, SIGMA, T, hedge=(n > 0)).error.std(ddof=1)
        assert std == np.float64(expected) or abs(std - expected) < 1e-4, (
            f"call n={n}: std {std:.6f} != {expected:.6f}"
        )


def test_digital_std_regression():
    paths = dh.crn_gbm_paths(X, R, SIGMA, T, list(DIGITAL_STD), N, rng=SEED)
    for n, expected in DIGITAL_STD.items():
        std = hg.hedge_digital(paths[n], K, R, SIGMA, T, hedge=(n > 0)).error.std(ddof=1)
        assert abs(std - expected) < 1e-4, f"digital n={n}: std {std:.6f} != {expected:.6f}"


def test_scaling_exponents_regression():
    # Call obeys ~1/sqrt(n) (exponent near -0.5); the digital clearly does NOT
    # (exponent near -0.25) -- a genuine, kept qualitative finding.
    paths = dh.crn_gbm_paths(X, R, SIGMA, T, list(CALL_STD), N, rng=SEED)
    call_std = [hg.hedge_call(paths[n], K, R, SIGMA, T).error.std(ddof=1)
                for n in [1, 3, 12, 84]]
    call_fit = an.fit_power_law([1, 3, 12, 84], call_std)
    assert call_fit["slope"] == np.float64(-0.4650) or abs(call_fit["slope"] + 0.465) < 5e-3

    pd_ = dh.crn_gbm_paths(X, R, SIGMA, T, list(DIGITAL_STD), N, rng=SEED)
    dig_std = [hg.hedge_digital(pd_[n], K, R, SIGMA, T).error.std(ddof=1)
               for n in [1, 3, 12, 84, 252]]
    dig_fit = an.fit_power_law([1, 3, 12, 84, 252], dig_std)
    assert abs(dig_fit["slope"] + 0.245) < 5e-3
    # The two regimes are clearly distinct.
    assert dig_fit["slope"] > call_fit["slope"] + 0.15
