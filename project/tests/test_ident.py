"""Isolation checks for Block D -- model_spec.md §10-20 (eq 26-45).

Targets and independent references:
  * discretize.zoh       (eq 31) vs scipy.signal.cont2discrete and vs the
                                  closed form A_c^{-1}(A-I)B_c
  * wls.wls              (eq 39-40) vs numpy.linalg.lstsq; Hessian PSD
  * huber.huber_fit      (eq 41) reduces to LS for delta -> inf; downweights
                                  an injected outlier
  * arx                  (eq 35-37) steady-state gain vs a simulated step
  * iv.iv_fit            (eq 42-43) OLS provably biased on closed-loop data,
                                  IV recovers the true gain
  * excitation           (eq 44) flags two near-collinear inputs
  * stability            (eq 45) agrees with numpy.linalg.eigvals
"""
import numpy as np
import pytest
from scipy import signal

from src.ident import arx, discretize, excitation, fir_ident, huber, iv, stability, wls

Ts = 600.0


# ------------------------------------------------------- eq 31
def test_zoh_matches_scipy_cont2discrete():
    Ac = np.array([[-1 / 3600.0, 0.0], [1 / 7200.0, -1 / 1800.0]])
    Bc = np.array([[1e-4], [0.0]])
    A, B = discretize.zoh(Ac, Bc, Ts)
    Ad, Bd, *_ = signal.cont2discrete((Ac, Bc, np.eye(2), np.zeros((2, 1))),
                                      Ts, method="zoh")
    assert np.allclose(A, Ad, atol=1e-12)
    assert np.allclose(B, Bd, atol=1e-12)


def test_zoh_matches_closed_form_for_invertible_Ac():
    Ac = np.array([[-2e-4, 1e-5], [0.0, -5e-4]])
    Bc = np.array([[1.0], [2.0]])
    A1, B1 = discretize.zoh(Ac, Bc, Ts)
    A2, B2 = discretize.zoh_invertible(Ac, Bc, Ts)
    assert np.allclose(A1, A2, atol=1e-12) and np.allclose(B1, B2, atol=1e-10)


def test_zoh_singular_Ac_still_works():
    """An integrator makes A_c singular; the block-exponential form must cope."""
    Ac = np.array([[0.0]])
    Bc = np.array([[1.0]])
    A, B = discretize.zoh(Ac, Bc, Ts)
    assert A[0, 0] == pytest.approx(1.0)
    assert B[0, 0] == pytest.approx(Ts)
    with pytest.raises(np.linalg.LinAlgError):
        discretize.zoh_invertible(Ac, Bc, Ts)


def test_zoh_guards():
    with pytest.raises(ValueError):
        discretize.zoh(np.zeros((2, 3)), np.zeros((2, 1)), Ts)
    with pytest.raises(ValueError):
        discretize.zoh(np.zeros((2, 2)), np.zeros((2, 1)), -1.0)


def test_zoh_full_splits_B_and_E():
    Ac = np.array([[-1e-4]])
    A, B, E = discretize.zoh_full(Ac, np.array([[1.0]]), np.array([[3.0]]), Ts)
    assert B.shape == (1, 1) and E.shape == (1, 1)
    assert E[0, 0] == pytest.approx(3.0 * B[0, 0])


# ------------------------------------------------------- eq 39-40
def test_wls_matches_lstsq_unregularised():
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(200, 4))
    theta = np.array([1.0, -2.0, 0.5, 3.0])
    Y = Phi @ theta + rng.normal(0, 0.1, 200)
    got = wls.wls(Phi, Y).theta
    ref, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    assert np.allclose(got, ref, atol=1e-10)


def test_wls_weights_reproduce_manual_scaling():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(60, 3))
    Y = rng.normal(size=60)
    w = rng.uniform(0.1, 3.0, 60)
    got = wls.wls(Phi, Y, weights=w).theta
    s = np.sqrt(w)[:, None]
    ref, *_ = np.linalg.lstsq(Phi * s, Y * np.sqrt(w), rcond=None)
    assert np.allclose(got, ref, atol=1e-10)


def test_wls_hessian_is_psd_and_ridge_shrinks_towards_theta0():
    rng = np.random.default_rng(2)
    Phi = rng.normal(size=(50, 3))
    Y = Phi @ np.array([2.0, 0.0, -1.0]) + rng.normal(0, 0.5, 50)
    H = wls.hessian(Phi, lam=0.7)
    assert np.all(np.linalg.eigvalsh((H + H.T) / 2) >= -1e-10)
    th0 = np.array([10.0, 10.0, 10.0])
    weak = wls.wls(Phi, Y, lam=1e-6, theta0=th0).theta
    strong = wls.wls(Phi, Y, lam=1e6, theta0=th0).theta
    assert np.linalg.norm(strong - th0) < np.linalg.norm(weak - th0)


def test_wls_guards():
    with pytest.raises(ValueError):
        wls.wls(np.zeros((3, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        wls.wls(np.zeros((3, 2)), np.zeros(3), weights=[-1, 1, 1])
    with pytest.raises(ValueError):
        wls.wls(np.empty((0, 2)), np.empty(0))


def test_freshness_weights_decay_with_age():
    w = wls.freshness_weights([0.0, 24.0, 48.0], tau_half_h=24.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(0.25)
    assert wls.freshness_weights([np.nan])[0] > 0


# ------------------------------------------------------- eq 41
def test_huber_reduces_to_least_squares_for_large_delta():
    rng = np.random.default_rng(3)
    Phi = rng.normal(size=(120, 3))
    Y = Phi @ np.array([1.0, -1.0, 2.0]) + rng.normal(0, 0.2, 120)
    ls = wls.wls(Phi, Y).theta
    hb = huber.huber_fit(Phi, Y, delta=1e9, sigma=1.0).theta
    assert np.allclose(ls, hb, atol=1e-8)


def test_huber_downweights_an_injected_outlier():
    rng = np.random.default_rng(4)
    Phi = rng.normal(size=(200, 2))
    theta = np.array([1.5, -0.5])
    Y = Phi @ theta + rng.normal(0, 0.05, 200)
    Y[7] += 200.0                              # a decoding blunder
    ls = wls.wls(Phi, Y).theta
    hb = huber.huber_fit(Phi, Y, delta=1.345).theta
    assert np.linalg.norm(hb - theta) < 0.1 * np.linalg.norm(ls - theta)


def test_huber_rho_psi_shapes():
    r = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    d = 1.0
    assert np.allclose(huber.huber_rho(r, d),
                       [3.0 - 0.5, 0.125, 0.0, 0.125, 3.0 - 0.5])
    assert np.allclose(huber.huber_psi(r, d), [-1, -0.5, 0, 0.5, 1])
    assert huber.huber_weight(np.array([0.0]), 1.0)[0] == 1.0
    assert huber.mad_sigma(np.ones(5)) > 0     # degenerate MAD floored


# ------------------------------------------------------- eq 35-37
def test_arx_steady_state_gain_matches_simulated_step():
    spec = arx.ArxSpec(na=1, inputs={"u": {"nb": 1, "delay": 2}})
    a1, b0 = 0.8, 0.3
    names = ["a1", "b[u,0]"]
    theta = np.array([a1, b0])
    n = 400
    u = {"u": np.ones(n)}
    y = arx.simulate_arx(theta, names, spec, u, n=n)
    K_eq37 = arx.steady_state_gain(theta, names, "u", spec.na)
    assert K_eq37 == pytest.approx(b0 / (1 - a1))
    assert y[-1] == pytest.approx(K_eq37, rel=1e-6)


def test_arx_gain_undefined_for_integrator():
    assert np.isnan(arx.steady_state_gain(np.array([1.0, 0.5]),
                                          ["a1", "b[u,0]"], "u", 1))


def test_arx_regressors_respect_delay_and_mask_nan():
    y = np.arange(10.0)
    u = {"u": np.arange(100.0, 110.0)}
    spec = arx.ArxSpec(na=1, inputs={"u": {"nb": 2, "delay": 3}})
    Phi, Y, names, valid = arx.build_arx_regressors(y, u, spec)
    assert names == ["a1", "b[u,0]", "b[u,1]"]
    assert not valid[:4].any() and valid[4:].all()
    assert Phi[4, 1] == pytest.approx(101.0)   # u_{k-3}
    assert Phi[4, 2] == pytest.approx(100.0)   # u_{k-4}
    with pytest.raises(ValueError):
        arx.build_arx_regressors(y, {"u": np.arange(3.0)}, spec)


def test_cross_correlation_recovers_a_planted_delay():
    rng = np.random.default_rng(5)
    u = rng.normal(size=3000)
    y = np.r_[np.zeros(7), u[:-7]] + rng.normal(0, 0.05, 3000)
    lag, _ = arx.cross_correlation_delay(u, y, max_lag=20)
    assert lag == 7


# ------------------------------------------------------- eq 42-43
def _closed_loop_dataset(n=25000, seed=7):
    """Proportional feedback on the *current* output:

        u_k = -Kc y_k + r_k,   y_k = a y_{k-1} + b u_k + eps_k

    so u_k contains eps_k and E[phi_k eps_k] != 0 -- eq 42 fails and OLS on
    (y_{k-1}, u_k) is inconsistent no matter how much data is collected.  The
    external set point r is the valid instrument of eq 43.
    """
    rng = np.random.default_rng(seed)
    a_true, b_true, Kc = 0.6, 0.4, 1.5
    r = rng.normal(0, 1.0, n)
    eps = rng.normal(0, 1.0, n)
    y = np.zeros(n)
    u = np.zeros(n)
    for k in range(1, n):
        # solve the algebraic loop y_k (1 + b Kc) = a y_{k-1} + b r_k + eps_k
        y[k] = (a_true * y[k - 1] + b_true * r[k] + eps[k]) / (1 + b_true * Kc)
        u[k] = -Kc * y[k] + r[k]
    return y, u, r, a_true, b_true


def test_ols_is_biased_and_iv_recovers_the_true_gain_closed_loop():
    y, u, r, a_true, b_true = _closed_loop_dataset()
    Phi = np.column_stack([np.r_[0.0, y[:-1]], u])[2:]
    Y = y[2:]
    ols, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    # instruments: the external set point and its lag (eq 43) -- exogenous
    Z = np.column_stack([r, np.r_[0.0, r[:-1]]])[2:]
    diag = iv.iv_diagnostics(Phi, Z)
    assert diag["relevant"], "instruments must be relevant (eq 43 rank)"
    res = iv.iv_fit(Phi, Y, Z)
    err_ols = abs(ols[1] - b_true)
    err_iv = abs(res.theta[1] - b_true)
    assert err_ols > 0.05, "the closed-loop dataset must actually bias OLS"
    assert err_iv < 0.02
    assert err_iv < 0.05 * err_ols
    # and IV *converges* (averaged over seeds, the sqrt(N) rate is visible)
    # while the OLS error stays at the same fixed bias.
    def mean_err(n):
        out = []
        for sd in range(5):
            yy, uu, rr, _, bb = _closed_loop_dataset(n=n, seed=100 + sd)
            P = np.column_stack([np.r_[0.0, yy[:-1]], uu])[2:]
            ZZ = np.column_stack([rr, np.r_[0.0, rr[:-1]]])[2:]
            out.append(abs(iv.iv_fit(P, yy[2:], ZZ).theta[1] - bb))
        return float(np.mean(out))

    assert mean_err(40_000) < mean_err(5_000)


def test_ols_bias_does_not_vanish_with_more_data():
    """Inconsistency, not variance: doubling N leaves the OLS error in place."""
    errs = []
    for n in (6000, 24000):
        y, u, r, a_true, b_true = _closed_loop_dataset(n=n, seed=11)
        Phi = np.column_stack([np.r_[0.0, y[:-1]], u])[2:]
        ols, *_ = np.linalg.lstsq(Phi, y[2:], rcond=None)
        errs.append(abs(ols[1] - b_true))
    assert errs[1] > 0.5 * errs[0]


def test_iv_requires_enough_instruments():
    with pytest.raises(ValueError):
        iv.iv_fit(np.zeros((10, 2)), np.zeros(10), np.zeros((10, 1)))


# ------------------------------------------------------- eq 44
def test_excitation_flags_near_collinear_inputs():
    rng = np.random.default_rng(8)
    x = rng.normal(size=500)
    Phi = np.column_stack([x, x + 1e-6 * rng.normal(size=500),
                           rng.normal(size=500)])
    rep = excitation.excitation_report(Phi, alpha=1e-3)
    assert not rep.persistently_exciting
    assert any(0 in g and 1 in g for g in rep.collinear_groups)
    Pk, names, rep2 = excitation.prune_collinear(Phi, ["a", "b", "c"])
    assert names == ["a", "c"] and rep2.dropped == [1]


def test_excitation_accepts_well_conditioned_design():
    rng = np.random.default_rng(9)
    Phi = rng.normal(size=(1000, 3))
    rep = excitation.excitation_report(Phi, alpha=1e-3)
    assert rep.full_rank and rep.persistently_exciting and not rep.collinear_groups


def test_excitation_drops_constant_column():
    Phi = np.column_stack([np.random.default_rng(0).normal(size=100),
                           np.ones(100)])
    _, names, rep = excitation.prune_collinear(Phi, ["x", "const"])
    assert names == ["x"] and 1 in rep.dropped
    assert excitation.excitation_report(np.empty((0, 0))).rank == 0


# ------------------------------------------------------- eq 45
def test_spectral_radius_agrees_with_numpy():
    rng = np.random.default_rng(10)
    A = rng.normal(size=(5, 5)) / 4
    assert stability.spectral_radius(A) == pytest.approx(
        float(np.max(np.abs(np.linalg.eigvals(A)))))
    assert stability.is_stable_discrete(A) == (stability.spectral_radius(A) < 1)


def test_arx_companion_and_projection():
    a = np.array([1.5, -0.4])          # rho > 1
    assert not stability.arx_is_stable(a)
    a_p = stability.project_to_stable(a, rho_max=0.95)
    assert stability.arx_is_stable(a_p)
    assert stability.spectral_radius(stability.arx_companion(a_p)) <= 0.9500001
    stable = np.array([0.5])
    assert np.allclose(stability.project_to_stable(stable), stable)


def test_continuous_stability_matches_eigenvalues():
    Ac = np.array([[-1.0, 2.0], [0.0, -3.0]])
    assert stability.is_stable_continuous(Ac)
    assert not stability.is_stable_continuous(np.array([[0.1]]))


# ------------------------------------------------------- FIR
def test_regularised_fir_recovers_a_smooth_impulse_response():
    rng = np.random.default_rng(11)
    n, L = 3000, 12
    g = np.exp(-np.arange(L) / 3.0)
    g /= g.sum()
    u = rng.normal(size=n)
    y = np.convolve(u, g)[:n] + rng.normal(0, 0.02, n)
    res, names = fir_ident.fit_fir({"u": u}, y, length=L, lam=1e-3)
    ghat = fir_ident.impulse_response(res.theta, names, "u")
    assert np.allclose(ghat, g, atol=0.02)
    assert sum(ghat) == pytest.approx(1.0, abs=0.03)   # eq 38 gain


def test_smoothness_matrix_shape():
    D = fir_ident.smoothness_matrix(4, 2)
    assert D.shape == (8, 8)
    with pytest.raises(ValueError):
        fir_ident.smoothness_matrix(0)
