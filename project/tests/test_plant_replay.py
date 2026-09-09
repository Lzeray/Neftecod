"""Isolation checks for the assembled predictor -- eq (51)-(52), (55).

Targets:
  * the offset stored on a laboratory row is the one available *before* that
    result (no leakage into the one-step forecast)
  * eta is held and P grows linearly between anchors (eq 52)
  * the chi^2 gate rejects an injected spike
  * a source hierarchy: with a fresh LIMS present, ПАК never anchors
"""
import numpy as np
import pandas as pd
import pytest

from src.kernels.sulfur_kernel import SulfurKernelParams
from src.predict.plant import PlantPredictor, replay


def _synthetic(n=200, lab_every=20, model_bias=3.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="10min")
    f = pd.DataFrame({"date": idx})
    truth = 8.0 + 0.5 * np.sin(np.arange(n) / 30.0)
    f["LIMS:sulfur"] = np.nan
    f["LIMS:sulfur__new"] = False
    f["LIMS:sulfur__age_h"] = np.nan
    for k in range(lab_every, n, lab_every):
        f.loc[k:, "LIMS:sulfur"] = truth[k]
        f.loc[k, "LIMS:sulfur__new"] = True
        f.loc[k:, "LIMS:sulfur__age_h"] = 0.0
    f["PAK:sulfur"] = truth + rng.normal(0, 0.2, n)
    f["PAK:sulfur__new"] = True
    f["PAK:sulfur__stale"] = False
    # A deliberately inert kernel (k_ref = 0 leaves C_out = C_in, so a
    # near-zero feed sulfur makes the kernel contribute ~0) -- the whole model
    # value is then the intercept, biased low by `model_bias`.
    f["LIMSFEED:sulfur"] = 1e-9
    f["derived:T_WABT_C"] = 365.0
    f["derived:pH2_proxy_MPa"] = 2.9
    f["derived:LHSV_rel"] = 1.0
    ut = pd.DataFrame({"c~lag": np.zeros(n)}, index=f.index)
    kern = SulfurKernelParams(k_ref=0.0, Ea=50_000.0, n=1.5, a=0.0,
                              T_ref_K=638.15)
    pred = PlantPredictor(kern, ["c~lag"], np.array([0.0]),
                          float(np.mean(truth) - model_bias), ["c"])
    return f, ut, pred, truth


def test_offset_removes_a_constant_model_bias():
    f, ut, pred, truth = _synthetic(model_bias=3.0)
    rr = replay(pred, f, ut, q_bias_per_h=0.05, r_lims=0.25, use_pak=False)
    r = rr.frame
    late = r.index > 60
    err = (r.loc[late, "y_hat"] - truth[late]).abs().mean()
    assert err < 1.0, "the LIMS-anchored offset must absorb the model bias"
    assert rr.n_anchors >= 5


def test_lab_row_forecast_does_not_use_its_own_result():
    """A perfect model with a huge bias: on the first lab row the prediction
    must still be wrong by the bias, because that result has not been used."""
    f, ut, pred, truth = _synthetic(model_bias=5.0, lab_every=25)
    rr = replay(pred, f, ut, q_bias_per_h=0.05, r_lims=0.01, use_pak=False)
    r = rr.frame
    first = int(np.where(r["LIMS:sulfur__new"].to_numpy(bool))[0][0])
    assert abs(r["y_hat"].iloc[first] - r["LIMS:sulfur"].iloc[first]) > 3.0
    # by the next lab row the offset has done its job
    second = int(np.where(r["LIMS:sulfur__new"].to_numpy(bool))[0][1])
    assert abs(r["y_hat"].iloc[second] - r["LIMS:sulfur"].iloc[second]) < 1.0


def test_offset_variance_grows_between_anchors_eq52():
    f, ut, pred, _ = _synthetic(lab_every=40)
    q = 0.1
    rr = replay(pred, f, ut, q_bias_per_h=q, r_lims=0.25, use_pak=False)
    r = rr.frame
    anchors = np.where(r["anchor_source"].to_numpy() != "")[0]
    a = int(anchors[0])
    seg = r["P_eta"].to_numpy()[a + 1:a + 20]
    assert np.all(np.diff(seg) > 0)
    # exactly linear in time, at rate q per hour
    dt = 10 / 60
    assert np.diff(seg).mean() == pytest.approx(q * dt, rel=1e-6)
    # and eta is constant over that stretch
    assert np.allclose(np.diff(r["eta"].to_numpy()[a + 1:a + 20]), 0.0)


def test_gate_rejects_an_injected_spike():
    f, ut, pred, _ = _synthetic(model_bias=0.0, lab_every=20)
    k = int(np.where(f["LIMS:sulfur__new"].to_numpy(bool))[0][3])
    f.loc[k, "LIMS:sulfur"] = 5000.0
    rr = replay(pred, f, ut, q_bias_per_h=0.01, r_lims=0.25, sigma_gate=6.0,
                use_pak=False)
    assert rr.n_rejected >= 1
    assert any(not e["accepted"] and e["k"] == k for e in rr.log)


def test_fresh_lims_wins_over_pak():
    f, ut, pred, _ = _synthetic(lab_every=20)
    rr = replay(pred, f, ut, q_bias_per_h=0.01, r_lims=0.25, use_pak=True)
    src = rr.frame["anchor_source"].to_numpy()
    lab_rows = np.where(f["LIMS:sulfur__new"].to_numpy(bool))[0]
    assert all(src[k] == "LIMS" for k in lab_rows if src[k])


def test_no_sources_leaves_the_offset_at_zero():
    f, ut, pred, _ = _synthetic()
    f["LIMS:sulfur__new"] = False
    rr = replay(pred, f, ut, use_pak=False)
    assert rr.n_anchors == 0
    assert np.allclose(rr.frame["eta"], 0.0)
    assert rr.frame["P_eta"].is_monotonic_increasing
