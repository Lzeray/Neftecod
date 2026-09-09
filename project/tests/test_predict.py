"""Isolation checks for Block F -- model_spec.md §26-28 (eq 55-60).

Targets:
  * multistep.build_maps (eq 56-57) -- the affine map must reproduce the
    step-by-step recursion exactly on a random stable system
  * uncertainty (eq 58) -- sigma grows with age and with distance from the
    training domain
  * uncertainty.set_error_bound (eq 59-60) -- the bound must actually contain
    simulated worst-case errors
  * greybox (eq 55) -- kernel + residual + offset; sign constraints hold
    out of domain
"""
import numpy as np
import pytest

from src.predict import greybox, multistep, uncertainty


def _random_stable(n=3, nu=2, nd=1, ny=1, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A *= 0.8 / max(np.abs(np.linalg.eigvals(A)).max(), 1e-9)
    return (A, rng.normal(size=(n, nu)), rng.normal(size=(ny, n)),
            rng.normal(size=(n, nd)), rng.normal(size=(ny, nu)),
            rng.normal(size=(ny, nd)))


# ------------------------------------------------------- eq 56-57
def test_affine_map_reproduces_the_recursion_exactly():
    A, B, C, E, D, F = _random_stable()
    H = 7
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=A.shape[0])
    U = rng.normal(size=(H, B.shape[1]))
    Dm = rng.normal(size=(H, E.shape[1]))
    maps = multistep.build_maps(A, B, C, E, D, F, H)
    Y_map = maps.predict(x0, U.ravel(), Dm.ravel())
    Y_rec = multistep.recursive_predict(A, B, C, x0, U.ravel(), Dm.ravel(),
                                        E, D, F, H)
    assert np.allclose(Y_map, Y_rec, atol=1e-12)


def test_affine_map_is_affine_in_U():
    A, B, C, E, D, F = _random_stable(seed=2)
    H = 5
    maps = multistep.build_maps(A, B, C, E, D, F, H)
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=A.shape[0])
    Dm = np.zeros(H * E.shape[1])
    U1 = rng.normal(size=H * B.shape[1])
    U2 = rng.normal(size=H * B.shape[1])
    lam = 0.3
    lhs = maps.predict(x0, lam * U1 + (1 - lam) * U2, Dm)
    rhs = lam * maps.predict(x0, U1, Dm) + (1 - lam) * maps.predict(x0, U2, Dm)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_maps_shapes_and_guards():
    A, B, C, E, D, F = _random_stable(n=2, nu=3, nd=2, ny=2, seed=4)
    maps = multistep.build_maps(A, B, C, E, D, F, H=4)
    assert maps.O.shape == (8, 2) and maps.Tu.shape == (8, 12)
    assert maps.Td.shape == (8, 8)
    with pytest.raises(ValueError):
        multistep.build_maps(A, B, C, H=0)
    with pytest.raises(ValueError):
        maps.predict(np.zeros(2), np.zeros(3), np.zeros(8))


def test_no_input_no_disturbance_is_the_free_response():
    A, B, C, E, D, F = _random_stable(seed=5)
    H = 6
    maps = multistep.build_maps(A, B, C, E, D, F, H)
    x0 = np.ones(A.shape[0])
    Y = maps.predict(x0, np.zeros(H * B.shape[1]), np.zeros(H * E.shape[1]))
    ref = np.concatenate([C @ np.linalg.matrix_power(A, h) @ x0
                          for h in range(1, H + 1)])
    assert np.allclose(Y, ref, atol=1e-12)


# ------------------------------------------------------- eq 58
def test_sigma_pred_grows_with_age_offset_and_horizon():
    um = uncertainty.UncertaintyModel(sigma_resid=1.0, c_tau=0.05,
                                      horizon_growth=0.01)
    base = um.sigma_pred()
    assert base == pytest.approx(1.0)
    assert um.sigma_pred(tau_h=10.0) > base
    assert um.sigma_pred(tau_h=20.0) > um.sigma_pred(tau_h=10.0)
    assert um.sigma_pred(P_eta=0.5) > base
    assert um.sigma_pred(h=5) > base
    assert um.sigma_pred(tau_h=10.0) == pytest.approx(np.sqrt(1.0 + 0.5))
    with pytest.raises(ValueError):
        um.sigma_pred(tau_h=-1.0)


def test_leverage_grows_away_from_the_training_domain():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(500, 3))
    mu, Sinv = uncertainty.fit_domain(X)
    um = uncertainty.UncertaintyModel(sigma_resid=1.0, x_mean=mu, x_cov_inv=Sinv)
    inside = um.sigma_pred(x=mu)
    near = um.sigma_pred(x=mu + 1.0)
    far = um.sigma_pred(x=mu + 8.0)
    assert inside == pytest.approx(1.0, abs=1e-9)
    assert far > near > inside
    with pytest.raises(ValueError):
        um.sigma_pred(x=np.zeros(2))


def test_z_quantile_values():
    assert uncertainty.z_quantile(0.05) == pytest.approx(1.6448536, rel=1e-6)
    assert uncertainty.z_quantile(0.01) == pytest.approx(2.3263479, rel=1e-6)
    with pytest.raises(ValueError):
        uncertainty.z_quantile(0.0)


# ------------------------------------------------------- eq 59-60
def test_set_bound_contains_simulated_worst_case_errors():
    A = np.array([[0.7, 0.1], [0.0, 0.5]])
    C = np.array([[1.0, 0.5]])
    w_half = np.array([0.2, 0.1])
    v_half = np.array([0.05])
    H = 8
    eps = uncertainty.set_error_bound(A, C, w_half, v_half, H)
    rng = np.random.default_rng(7)
    worst = np.zeros((H, 1))
    for _ in range(4000):
        e = np.zeros(2)
        for h in range(H):
            w = rng.uniform(-w_half, w_half)
            e = A @ e + w
            v = rng.uniform(-v_half, v_half)
            worst[h, 0] = max(worst[h, 0], float((C @ e + v)[0]))
    assert np.all(worst[:, 0] <= eps[:, 0] + 1e-12)
    # and the bound is not absurdly loose: random sampling reaches a fair share
    assert worst[0, 0] > 0.3 * eps[0, 0]


def test_set_bound_monotone_and_guards():
    A = np.array([[0.9]]); C = np.array([[1.0]])
    eps = uncertainty.set_error_bound(A, C, [0.1], [0.02], 5)
    assert np.all(np.diff(eps[:, 0]) > 0)
    zero = uncertainty.set_error_bound(A, C, [0.0], [0.0], 3)
    assert np.allclose(zero, 0.0)
    with pytest.raises(ValueError):
        uncertainty.set_error_bound(A, C, [-0.1], [0.0], 3)
    with pytest.raises(ValueError):
        uncertainty.set_error_bound(A, C, [0.1, 0.1], [0.0], 3)


def test_empirical_half_width():
    r = np.array([-1.0, 0.5, 2.0, np.nan])
    assert uncertainty.empirical_half_widths(r, 1.0) == pytest.approx(2.0)
    assert uncertainty.empirical_half_widths(np.array([])) == 0.0


# ------------------------------------------------------- eq 55
def test_greybox_is_kernel_plus_residual_plus_offset():
    rng = np.random.default_rng(8)
    n = 400
    X = rng.normal(size=(n, 2))
    kernel = 5.0 + 0.3 * X[:, 0]
    y = kernel + (0.7 * X[:, 1] - 0.2 * X[:, 0]) + rng.normal(0, 0.05, n)
    m = greybox.fit_greybox(X, y, kernel, ["T", "q"])
    assert m.sensitivity("q") == pytest.approx(0.7, abs=0.05)
    pred = m.predict(kernel, X, eta=0.0)
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    # the offset shifts the prediction by exactly eta
    assert np.allclose(m.predict(kernel, X, eta=2.5) - pred, 2.5)


def test_sign_constraint_holds_out_of_domain():
    """Physics: d S / d T_WABT <= 0.  Even if the data suggest otherwise, the
    fitted model must never claim that heating the reactor raises sulfur."""
    rng = np.random.default_rng(9)
    n = 300
    X = rng.normal(size=(n, 1))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.01, n)      # wrong sign on purpose
    cons = [greybox.SignConstraint("T_WABT", -1, "HDS kinetics, eq 13-15")]
    m = greybox.fit_greybox(X, y, None, ["T_WABT"], sign_constraints=cons)
    assert m.sensitivity("T_WABT") <= 0.0
    assert m.fit_report["sign_violations"], "the violation must be reported"
    far = np.array([[100.0], [-100.0]])             # far outside the domain
    d = m.residual(far)
    assert d[0] <= d[1] + 1e-12                     # still non-increasing in T


def test_greybox_iv_path_reduces_closed_loop_bias():
    rng = np.random.default_rng(10)
    n = 6000
    r = rng.normal(size=n)
    eps = rng.normal(0, 1.0, n)
    b_true, Kc = 0.6, 1.2
    y = (b_true * r + eps) / (1 + b_true * Kc)
    u = -Kc * y + r
    X = u.reshape(-1, 1)
    ols = greybox.fit_greybox(X, y, None, ["u"], lam=0.0, huber_delta=1e9)
    ivm = greybox.fit_greybox(X, y, None, ["u"], instruments=r.reshape(-1, 1),
                              lam=0.0, huber_delta=1e9)
    assert abs(ivm.sensitivity("u") - b_true) < abs(ols.sensitivity("u") - b_true)


def test_greybox_guards():
    with pytest.raises(ValueError):
        greybox.fit_greybox(np.zeros((2, 3)), np.zeros(2), None)
    m = greybox.GreyBoxModel(["a"], np.array([1.0]))
    with pytest.raises(ValueError):
        m.residual(np.zeros((3, 2)))


# ------------------------------------------------------- §26 shaped inputs
def test_kernel_reads_shaped_inputs_not_instantaneous_ones():
    """model_spec.md §26: g_phi must see the delayed/shaped inputs of eq 32-33.

    Feeding it the instantaneous measurement would claim that a set-point
    change reaches the product immediately, when identification puts the
    response hours later.
    """
    import pandas as pd

    from src.ident.channels import KIN
    from src.kernels.sulfur_kernel import SulfurKernelParams
    from src.predict.plant import PlantPredictor

    kern = SulfurKernelParams(k_ref=0.4, Ea=100_000.0, n=1.5, a=1.0,
                              T_ref_K=273.15 + 366.0)
    p = PlantPredictor(kern, [], np.zeros(0), 0.0, [])
    frame = pd.DataFrame({
        "LIMSFEED:sulfur": [9000.0], "derived:T_WABT_C": [400.0],
        "derived:pH2_proxy_MPa": [2.9], "derived:LHSV_rel": [1.0]})
    ut = pd.DataFrame({f"{KIN}S_in": [9000.0], f"{KIN}T_WABT": [360.0],
                       f"{KIN}pH2": [2.9], f"{KIN}LHSV": [1.0]})
    shaped = p.kernel_series(frame, ut)
    raw = p.kernel_series(frame)
    assert shaped[0] != pytest.approx(raw[0]), \
        "the shaped and instantaneous inputs must not give the same answer"
    # the shaped input is 40 degC colder, so it must predict MORE sulfur
    assert shaped[0] > raw[0]


def test_horizon_response_is_delayed_and_saturating():
    """A held move must show no effect before the transport delay (eq 32-33)."""
    import pandas as pd

    from src.ident.channels import KIN
    from src.ident.channels import ChannelDynamics
    from src.kernels.sulfur_kernel import SulfurKernelParams
    from src.predict.plant import PlantPredictor, horizon_response

    Ts = 600.0
    kern = SulfurKernelParams(k_ref=0.4, Ea=100_000.0, n=1.5, a=1.0,
                              T_ref_K=273.15 + 366.0)
    p = PlantPredictor(kern, [], np.zeros(0), 0.0, [])
    row = pd.Series({"HDS:F26": 250.0})
    ut = pd.Series({f"{KIN}S_in": 9000.0, f"{KIN}T_WABT": 366.0,
                    f"{KIN}pH2": 2.9, f"{KIN}LHSV": 1.0})
    dyn = {"HDS:T5": ChannelDynamics(key="HDS:T5", delay_samples=12,
                                     tau_samples=6.0, peak_corr=0.2,
                                     n_obs=500, Ts_s=Ts)}
    hr = horizon_response(p, row, ut, "HDS:T5", 5.0, dyn, Ts, H=48)
    assert hr["available"]
    traj = np.array(hr["trajectory_mgkg"])
    # nothing happens during the 12-sample (2 h) transport delay
    assert np.allclose(traj[:12], hr["y0_mgkg"], atol=1e-9)
    assert traj[-1] < hr["y0_mgkg"]                 # hotter -> less sulfur
    assert hr["delta_at_horizon_mgkg"] < 0
    assert hr["time_to_63pct_h"] >= hr["delay_h"]
    # a channel with no kernel input returns "unavailable" rather than zero
    assert not horizon_response(p, row, ut, "AVT:F29", 1.0, dyn, Ts, 10)[
        "available"]
