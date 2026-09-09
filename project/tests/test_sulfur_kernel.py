"""Isolation checks for the sulfur kernel -- eq (2), (14), (16).

Targets: monotonicity in every argument, recovery of planted parameters,
robustness of the fit to injected off-spec lab results, and refusal of a
degenerate data set.
"""
import numpy as np
import pytest

from src.kernels import sulfur_kernel as SK

P = SK.SulfurKernelParams(k_ref=0.5, Ea=90_000.0, n=1.5, a=1.0,
                          T_ref_K=273.15 + 365.0)


def test_prediction_is_monotone_in_every_argument():
    base = float(SK.predict_sulfur(9000.0, 365.0, 2.9, 1.0, P))
    assert 0 < base < 9000.0
    assert SK.predict_sulfur(9000.0, 375.0, 2.9, 1.0, P) < base   # hotter
    assert SK.predict_sulfur(9000.0, 365.0, 3.2, 1.0, P) < base   # more H2
    assert SK.predict_sulfur(9000.0, 365.0, 2.9, 1.2, P) > base   # faster feed
    assert SK.predict_sulfur(11000.0, 365.0, 2.9, 1.0, P) > base  # dirtier feed


def test_sensitivities_have_the_physically_required_signs():
    assert float(SK.dS_dT(9000.0, 365.0, 2.9, 1.0, P)) < 0
    assert float(SK.dS_dSin(9000.0, 365.0, 2.9, 1.0, P)) > 0
    assert float(SK.dS_dlhsv(9000.0, 365.0, 2.9, 1.0, P)) > 0


def test_fit_recovers_planted_parameters():
    rng = np.random.default_rng(0)
    n = 600
    S_in = rng.uniform(7000, 11000, n)
    T = rng.uniform(355, 380, n)
    pH2 = rng.uniform(2.6, 3.1, n)
    lh = rng.uniform(0.85, 1.15, n)
    y = SK.predict_sulfur(S_in, T, pH2, lh, P)
    y = y * np.exp(rng.normal(0, 0.02, n))
    got = SK.fit_sulfur_kernel(S_in, T, pH2, lh, y, T_ref_C=365.0)
    pred = SK.predict_sulfur(S_in, T, pH2, lh, got)
    # parameters are only jointly identifiable; check the *predictions* match
    assert np.corrcoef(np.log(pred), np.log(y))[0, 1] > 0.98
    assert float(np.mean(np.abs(pred / y - 1.0))) < 0.05
    assert 1.0 < got.n < 2.0


def test_fit_is_robust_to_injected_off_spec_results():
    rng = np.random.default_rng(1)
    n = 500
    S_in = rng.uniform(7000, 11000, n)
    T = rng.uniform(355, 380, n)
    pH2 = rng.uniform(2.6, 3.1, n)
    lh = rng.uniform(0.85, 1.15, n)
    y = SK.predict_sulfur(S_in, T, pH2, lh, P)
    dirty = y.copy()
    dirty[:15] = 2120.0                      # the real record contains these
    clean_fit = SK.fit_sulfur_kernel(S_in, T, pH2, lh, y, T_ref_C=365.0)
    dirty_fit = SK.fit_sulfur_kernel(S_in, T, pH2, lh, dirty, T_ref_C=365.0)
    pc = SK.predict_sulfur(S_in, T, pH2, lh, clean_fit)
    pd_ = SK.predict_sulfur(S_in, T, pH2, lh, dirty_fit)
    assert float(np.median(np.abs(pd_ / pc - 1.0))) < 0.25


def test_fit_refuses_a_degenerate_dataset():
    with pytest.raises(ValueError):
        SK.fit_sulfur_kernel([1.0] * 5, [360.0] * 5, [2.9] * 5, [1.0] * 5,
                             [8.0] * 5)


def test_prediction_is_nan_safe():
    out = SK.predict_sulfur([np.nan, 9000.0], [365.0, 365.0], [2.9, 2.9],
                            [1.0, 1.0], P)
    assert np.isnan(out[0]) and np.isfinite(out[1])
