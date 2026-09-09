"""Isolation checks for Block B -- model_spec.md §12-13, §15 (eq 32-34, 38).

Targets:
  * lag.pure_delay / delay_states  -- eq 34, exact pure delay as extra states
  * lag.first_order_filter / foptd -- eq 33, verified against the closed-form
    step response (eq 33 upper branch) and against scipy.signal.lsim
  * fir.fir_weights / apply_fir    -- non-negative unit-sum kernel, eq 38 gain
"""
import numpy as np
import pytest
from scipy import signal

from src.features import fir, lag

Ts = 600.0


# ---------------------------------------------------------------- eq 34
def test_pure_delay_exact():
    u = np.arange(6.0)
    d = lag.pure_delay(u, 2)
    assert np.isnan(d[:2]).all()
    assert list(d[2:]) == [0.0, 1.0, 2.0, 3.0]
    assert list(lag.pure_delay(u, 0)) == list(u)


def test_delay_states_reproduce_pure_delay():
    u = np.arange(10.0)
    Z = lag.delay_states(u, 4)
    assert Z.shape == (10, 4)
    # column n-1 IS u_{k-n}; a linear model on it is exactly a pure delay
    assert np.allclose(Z[4:, 3], lag.pure_delay(u, 4)[4:])
    assert lag.delay_states(u, 0).shape == (10, 0)


def test_pure_delay_rejects_negative():
    with pytest.raises(ValueError):
        lag.pure_delay(np.arange(3.0), -1)


# ---------------------------------------------------------------- eq 33
def test_first_order_step_matches_closed_form():
    tau = 3600.0
    n = 60
    u = np.ones(n)
    z = lag.first_order_filter(u, tau, Ts, z0=0.0, gain=2.0)
    t = np.arange(n) * Ts
    # z[k] is the state *before* absorbing u[k]; the closed form after k steps
    # of a unit step is K(1 - a^k)
    a = np.exp(-Ts / tau)
    closed = 2.0 * (1.0 - a ** np.arange(n))
    assert np.allclose(z, closed, atol=1e-12)
    # and it converges to the static gain
    assert z[-1] == pytest.approx(2.0, abs=1e-3)


def test_first_order_matches_scipy_zoh_discretisation():
    """Independent reference: scipy.signal.cont2discrete on K/(tau s + 1)."""
    tau, K = 5400.0, 1.7
    sysd = signal.cont2discrete(([K], [tau, 1.0]), Ts, method="zoh")
    num, den = sysd[0].ravel(), sysd[1].ravel()
    rng = np.random.default_rng(3)
    u = rng.normal(size=200)
    ref = signal.lfilter(num, den, u)
    got = lag.first_order_filter(u, tau, Ts, gain=K)
    # eq 33 is exactly the ZOH discretisation of K/(tau s + 1) with the strictly
    # causal numerator scipy produces (num = [0, K(1-a)], den = [1, -a]).
    assert np.allclose(got, ref, atol=1e-12)


def test_foptd_delay_plus_lag():
    tau, theta = 3600.0, 3 * Ts
    u = np.ones(40)
    z = lag.foptd(u, tau, theta, Ts, gain=1.0)
    assert np.allclose(z[:4], 0.0)             # 3 samples of delay + z0
    assert z[10] > 0 and z[-1] == pytest.approx(1.0, abs=1e-2)
    t = np.arange(40) * Ts
    closed = lag.foptd_step_response(t, 1.0, tau, theta)
    assert np.allclose(z, closed, atol=1e-12)


def test_first_order_degenerates_and_guards():
    u = np.arange(5.0)
    assert np.allclose(lag.first_order_filter(u, 0.0, Ts, gain=3.0), 3.0 * u)
    with pytest.raises(ValueError):
        lag.first_order_filter(u, 100.0, 0.0)


def test_first_order_handles_nan_without_propagating():
    u = np.array([1.0, np.nan, 1.0, 1.0])
    z = lag.first_order_filter(u, 3600.0, Ts)
    assert np.isfinite(z).all()


# ---------------------------------------------------------------- eq 38
def test_fir_weights_non_negative_unit_sum():
    for kind in ("exp", "box"):
        w = fir.fir_weights(12, 4.0, kind)
        assert (w >= 0).all()
        assert w.sum() == pytest.approx(1.0)
    w0 = fir.fir_weights(5, 0.0, "exp")
    assert w0[0] == 1.0 and w0[1:].sum() == 0.0
    with pytest.raises(ValueError):
        fir.fir_weights(0, 1.0)


def test_apply_fir_hand_example_and_alignment():
    u = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.5, 0.5])
    out = fir.apply_fir(u, w)
    assert np.isnan(out[0])
    assert list(out[1:]) == [1.5, 2.5, 3.5]
    # a unit-sum kernel preserves the steady state
    const = fir.apply_fir(np.full(20, 7.0), fir.fir_weights(6, 2.0))
    assert const[-1] == pytest.approx(7.0)


def test_fir_regressor_matrix_and_gain():
    u = np.arange(6.0)
    X = fir.fir_regressor_matrix(u, 3)
    assert X.shape == (6, 3)
    assert np.allclose(X[2:, 2], u[:-2])
    assert fir.steady_state_gain_fir([0.5, 0.3, 0.2]) == pytest.approx(1.0)


def test_apply_fir_shorter_than_kernel():
    assert np.isnan(fir.apply_fir(np.arange(2.0), np.ones(5) / 5)).all()


def test_delay_and_first_order_lag_commute():
    """The channel search relies on foptd(u, tau, n Ts) == shift(filter(u), n).

    Both blocks are LTI, so they commute; the identification loop uses that to
    evaluate the filter once per tau instead of once per (tau, delay) pair.
    """
    rng = np.random.default_rng(12)
    u = rng.normal(size=500)
    tau, n = 5400.0, 7
    a = lag.foptd(u, tau, n * Ts, Ts)
    b = lag.pure_delay(lag.first_order_filter(u, tau, Ts), n, fill=0.0)
    assert np.allclose(a[n:], b[n:], atol=1e-10)
