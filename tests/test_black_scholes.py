"""Correctness tests for the Black-Scholes pricing and Greeks: put-call parity
and every Greek cross-checked against central finite differences."""

import numpy as np
import pytest

from deltahedge import black_scholes as bs

R, SIGMA, K, T = 0.10, 0.20, 90.0, 0.25
SPOTS = np.array([70.0, 90.0, 100.0, 130.0])


def fd(fn, S, arg="S", h=1e-4, order=1):
    """Central finite-difference of ``fn`` wrt spot (order 1 or 2)."""
    if order == 1:
        return (fn(S + h) - fn(S - h)) / (2 * h)
    return (fn(S + h) - 2 * fn(S) + fn(S - h)) / h**2


def test_known_baseline_values():
    # Audit-confirmed baseline (S0=100, K=90).
    assert bs.call_price(100.0, K, R, SIGMA, T) == pytest.approx(12.645, abs=1e-3)
    assert bs.call_delta(100.0, K, R, SIGMA, T) == pytest.approx(0.9121, abs=1e-3)


def test_put_call_parity():
    C = bs.call_price(SPOTS, K, R, SIGMA, T)
    P = bs.put_price(SPOTS, K, R, SIGMA, T)
    np.testing.assert_allclose(C - P, SPOTS - K * np.exp(-R * T), atol=1e-10)


def test_call_delta_matches_fd():
    analytic = bs.call_delta(SPOTS, K, R, SIGMA, T)
    numeric = fd(lambda S: bs.call_price(S, K, R, SIGMA, T), SPOTS, order=1)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_call_gamma_matches_fd():
    analytic = bs.call_gamma(SPOTS, K, R, SIGMA, T)
    numeric = fd(lambda S: bs.call_price(S, K, R, SIGMA, T), SPOTS, order=2)
    np.testing.assert_allclose(analytic, numeric, atol=1e-3)


def test_call_vega_matches_fd():
    h = 1e-5
    numeric = (
        bs.call_price(SPOTS, K, R, SIGMA + h, T)
        - bs.call_price(SPOTS, K, R, SIGMA - h, T)
    ) / (2 * h)
    np.testing.assert_allclose(bs.call_vega(SPOTS, K, R, SIGMA, T), numeric, atol=1e-3)


def test_call_theta_matches_fd():
    # theta = dV/dt = -dV/dtau ; bump tau, convert sign.
    h = 1e-5
    numeric = -(
        bs.call_price(SPOTS, K, R, SIGMA, T + h)
        - bs.call_price(SPOTS, K, R, SIGMA, T - h)
    ) / (2 * h)
    np.testing.assert_allclose(bs.call_theta(SPOTS, K, R, SIGMA, T), numeric, atol=1e-3)


def test_digital_delta_matches_fd():
    analytic = bs.digital_delta(SPOTS, K, R, SIGMA, T)
    numeric = fd(lambda S: bs.digital_price(S, K, R, SIGMA, T), SPOTS, order=1)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_digital_gamma_matches_fd():
    analytic = bs.digital_gamma(SPOTS, K, R, SIGMA, T)
    numeric = fd(lambda S: bs.digital_price(S, K, R, SIGMA, T), SPOTS, order=2)
    np.testing.assert_allclose(analytic, numeric, atol=1e-3)


def test_terminal_limits():
    # tau -> 0: price collapses to intrinsic / indicator.
    np.testing.assert_allclose(
        bs.call_price(SPOTS, K, R, SIGMA, 0.0), np.maximum(SPOTS - K, 0.0)
    )
    np.testing.assert_allclose(
        bs.digital_price(SPOTS, K, R, SIGMA, 0.0), (SPOTS >= K).astype(float)
    )
    np.testing.assert_allclose(
        bs.call_delta(SPOTS, K, R, SIGMA, 0.0), (SPOTS > K).astype(float)
    )


def test_vectorisation_matches_scalar():
    vec = bs.call_price(SPOTS, K, R, SIGMA, T)
    scal = np.array([bs.call_price(float(s), K, R, SIGMA, T) for s in SPOTS])
    np.testing.assert_allclose(vec, scal)
