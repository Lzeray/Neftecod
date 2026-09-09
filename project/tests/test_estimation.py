"""Isolation checks for Block E -- model_spec.md §22-24 (eq 47-52).

Targets:
  * offset.augment_offset (eq 47) -- observability of the augmented system
  * kalman.predict/update (eq 48-50) -- error covariance vs Monte Carlo;
    no measurement -> pure prediction with growing P; chi^2 gate rejects a
    stuck/biased sensor and accepts valid ones
  * fusion.BiasFilter (eq 51-52) -- LIMS anchoring, variance growth between
    updates, LIMS > ПАК > ВАК hierarchy
"""
import numpy as np
import pytest

from src.estimation import fusion, gating, kalman, offset
from src.estimation.fusion import SourceRank, SourceReading


# ------------------------------------------------------- eq 47
def test_augmented_model_shapes_and_integrator():
    A = np.array([[0.9]]); B = np.array([[0.1]]); E = np.array([[0.05]])
    C = np.array([[1.0]]); D = np.array([[0.0]]); F = np.array([[0.0]])
    m = offset.augment_offset(A, B, E, C, D, F)
    assert m.A.shape == (2, 2) and m.C.shape == (1, 2)
    assert m.A[1, 1] == 1.0 and m.A[1, 0] == 0.0      # eta is a random walk
    assert m.n_eta == 1 and m.n == 2


def test_augmentation_refuses_more_offsets_than_measurements():
    A = np.eye(2); B = np.zeros((2, 1)); E = np.zeros((2, 1))
    C = np.array([[1.0, 0.0]]); D = np.zeros((1, 1)); F = np.zeros((1, 1))
    with pytest.raises(ValueError):
        offset.augment_offset(A, B, E, C, D, F, C_eta=np.eye(1)[:, [0, 0]])


def test_augmentation_refuses_unobservable_pair():
    A = np.array([[0.9]]); C = np.array([[1.0]])
    z1 = np.zeros((1, 1))
    # B_eta chosen so that the offset is indistinguishable from the state
    with pytest.raises(ValueError):
        offset.augment_offset(np.eye(2), np.zeros((2, 1)), np.zeros((2, 1)),
                              np.array([[1.0, 0.0]]), z1, z1)
    assert offset.is_observable(A, C)


# ------------------------------------------------------- eq 48-50
def test_no_measurement_reduces_to_prediction_and_P_grows():
    A = np.array([[0.95]]); Q = np.array([[0.1]])
    x, P = np.array([1.0]), np.array([[1.0]])
    xp, Pp = kalman.predict(x, P, A, Q)
    assert xp[0] == pytest.approx(0.95)
    assert Pp[0, 0] == pytest.approx(0.95 ** 2 + 0.1)
    x2, P2, info = kalman.update(xp, Pp, np.array([np.nan]), np.array([[1.0]]),
                                 np.array([[1.0]]))
    assert not info["accepted"] and np.allclose(x2, xp) and np.allclose(P2, Pp)
    # over several blind steps P keeps growing
    Ps = [P]
    for _ in range(5):
        _, Pn = kalman.predict(x, Ps[-1], A, Q)
        Ps.append(Pn)
    assert all(Ps[i + 1][0, 0] > Ps[i][0, 0] for i in range(3))


def test_error_covariance_matches_monte_carlo():
    """Linear-Gaussian system: the reported P must match the empirical error
    covariance of an independent Monte-Carlo simulation."""
    rng = np.random.default_rng(0)
    A = np.array([[0.9]]); C = np.array([[1.0]])
    Q = np.array([[0.05]]); R = np.array([[0.2]])
    n_steps, n_mc = 60, 4000
    errs = np.zeros((n_mc, n_steps))
    P_reported = None
    for m in range(n_mc):
        x_true = np.array([0.0])
        x, P = np.array([0.0]), np.array([[1.0]])
        for k in range(n_steps):
            x_true = A @ x_true + rng.normal(0, np.sqrt(Q[0, 0]), 1)
            y = C @ x_true + rng.normal(0, np.sqrt(R[0, 0]), 1)
            xp, Pp = kalman.predict(x, P, A, Q)
            x, P, _ = kalman.update(xp, Pp, y, C, R, gate_p=None)
            errs[m, k] = float(x_true[0] - x[0])
        P_reported = P
    empirical = float(np.var(errs[:, -1]))
    assert P_reported[0, 0] == pytest.approx(empirical, rel=0.08)


def test_gate_rejects_a_biased_sensor_and_accepts_a_valid_one():
    A = np.array([[1.0]]); C = np.array([[1.0]])
    Q = np.array([[0.01]]); R = np.array([[0.04]])
    x, P = np.array([8.0]), np.array([[0.05]])
    xp, Pp = kalman.predict(x, P, A, Q)
    _, _, ok = kalman.update(xp, Pp, np.array([8.1]), C, R, gate_p=0.99)
    assert ok["accepted"]
    _, _, bad = kalman.update(xp, Pp, np.array([40.0]), C, R, gate_p=0.99)
    assert not bad["accepted"] and "gate" in bad["reason"]


def test_stale_flag_blocks_update_even_when_consistent():
    """A frozen analyser can agree perfectly with the prediction -- §23."""
    A = np.array([[1.0]]); C = np.array([[1.0]])
    Q = np.array([[0.01]]); R = np.array([[0.04]])
    xp, Pp = kalman.predict(np.array([8.0]), np.array([[0.05]]), A, Q)
    x, P, info = kalman.update(xp, Pp, np.array([8.0]), C, R, stale=True)
    assert not info["accepted"] and "stale" in info["reason"]
    assert np.allclose(x, xp) and np.allclose(P, Pp)


def test_joseph_form_keeps_P_symmetric_psd():
    rng = np.random.default_rng(1)
    A = np.array([[0.8, 0.1], [0.0, 0.95]])
    C = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.02
    R = np.array([[0.3]])
    x, P = np.zeros(2), np.eye(2)
    for _ in range(200):
        x, P = kalman.predict(x, P, A, Q)
        x, P, _ = kalman.update(x, P, np.array([rng.normal()]), C, R, gate_p=None)
        assert np.allclose(P, P.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(P) > -1e-12)


def test_run_filter_counts_rejections():
    A = np.array([[1.0]]); C = np.array([[1.0]])
    ys = [np.array([0.0])] * 10 + [np.array([50.0])] + [None]
    st = kalman.run_filter(A, None, C, np.array([[0.001]]), np.array([[0.01]]),
                           ys, gate_p=0.99, x0=np.array([0.0]),
                           P0=np.array([[0.01]]))
    assert st.n_rejected >= 1 and st.n_updates >= 9


# ------------------------------------------------------- eq 51-52
def test_bias_filter_anchors_to_lims_and_shrinks_variance():
    bf = fusion.BiasFilter(eta=0.0, P=4.0, q_per_h=0.02)
    info = bf.update(t_h=0.0, y_meas=9.0, y_model=8.0, R=0.25)
    K_expected = 4.0 / (4.0 + 0.25)
    assert info["K"] == pytest.approx(K_expected)
    assert bf.eta == pytest.approx(K_expected * 1.0)
    assert bf.P == pytest.approx((1 - K_expected) * 4.0)


def test_bias_variance_grows_between_updates_eq52():
    bf = fusion.BiasFilter(eta=0.5, P=0.1, q_per_h=0.02)
    bf.update(0.0, 9.0, 9.0, 0.25)
    P0 = bf.P
    eta_t, P_t = bf.value(10.0)
    assert eta_t == pytest.approx(bf.eta)             # eta is held (eq 52)
    assert P_t == pytest.approx(P0 + 0.02 * 10.0)     # variance grows
    assert bf.value(0.0)[1] == pytest.approx(P0)
    with pytest.raises(ValueError):
        bf.propagate(-1.0)


def test_pak_moves_the_offset_less_than_lims():
    """R_PAK >> R_lab, so the same discrepancy moves eta far less (§24)."""
    lab = fusion.BiasFilter(P=1.0); pak = fusion.BiasFilter(P=1.0)
    lab.update(0.0, 9.0, 8.0, R=0.25, source=SourceRank.LIMS)
    pak.update(0.0, 9.0, 8.0, R=4.0, source=SourceRank.PAK)
    assert abs(pak.eta) < abs(lab.eta)


def test_hierarchy_prefers_fresh_lims_over_pak_and_vak():
    readings = [
        SourceReading("VAK", SourceRank.VAK, 7.0, 0.0, 9.0),
        SourceReading("PAK", SourceRank.PAK, 8.5, 0.2, 4.0),
        SourceReading("LIMS", SourceRank.LIMS, 9.1, 3.0, 0.25),
    ]
    limits = {"LIMS": 72.0, "PAK": 2.0, "VAK": np.inf}
    win, trail = fusion.select_source(readings, limits)
    assert win.name == "LIMS"
    assert [t["source"] for t in trail][0] == "LIMS"


def test_stale_pak_is_skipped_and_old_lims_falls_through():
    readings = [
        SourceReading("LIMS", SourceRank.LIMS, 9.1, 100.0, 0.25),   # too old
        SourceReading("PAK", SourceRank.PAK, 8.5, 0.2, 4.0, stale=True),
        SourceReading("VAK", SourceRank.VAK, 7.0, 0.0, 9.0),
    ]
    limits = {"LIMS": 72.0, "PAK": 2.0, "VAK": np.inf}
    win, trail = fusion.select_source(readings, limits)
    assert win.name == "VAK"
    reasons = {t["source"]: t["decision"] for t in trail}
    assert "too old" in reasons["LIMS"] and reasons["PAK"] == "not usable"


def test_fuse_reports_lims_pak_conflict_and_resolves_by_lims():
    bf = fusion.BiasFilter(P=1.0)
    readings = [
        SourceReading("LIMS", SourceRank.LIMS, 9.5, 1.0, 0.25),
        SourceReading("PAK", SourceRank.PAK, 6.0, 0.1, 4.0),
    ]
    out = fusion.fuse(readings, y_model=8.0, bias=bf, t_h=1.0,
                      max_age_h={"LIMS": 72.0, "PAK": 2.0})
    assert out["source"] == "LIMS" and out["anchored"]
    assert out["conflict"]["gap"] == pytest.approx(3.5)
    assert "LIMS" in out["conflict"]["resolved_by"]


def test_fuse_without_any_source_holds_the_offset():
    bf = fusion.BiasFilter(eta=1.2, P=0.5, q_per_h=0.02)
    bf.update(0.0, 9.0, 9.0, 0.25)
    out = fusion.fuse([], y_model=8.0, bias=bf, t_h=5.0, max_age_h={})
    assert out["source"] is None and not out["anchored"]
    assert out["y_fused"] == pytest.approx(8.0 + bf.eta)
    assert out["P_eta"] > bf.P


# ------------------------------------------------------- gating
def test_gate_scalar_rules():
    assert gating.gate_scalar(8.0, 8.1, 0.05, 0.04).accept
    assert not gating.gate_scalar(np.nan, 8.0, 0.05, 0.04).accept
    assert not gating.gate_scalar(8.0, 8.0, 0.05, 0.04, stale=True).accept
    d = gating.gate_scalar(-5.0, 8.0, 0.05, 0.04, valid_range=(0.0, 50.0))
    assert not d.accept and "physical range" in d.reason
    assert not gating.gate_scalar(40.0, 8.0, 0.05, 0.04).accept
