"""Isolation checks for Block G -- model_spec.md §29-30 (eq 61-62).

Targets:
  * severity.SeverityTerm/SeverityIndex (eq 61) -- zero inside the limits,
    monotone beyond them, robust standardisation insensitive to outliers
  * deactivation (eq 62) -- detects an injected slow upward T_WABT drift at
    fixed quality and reports it as a flagged hypothesis
"""
import numpy as np
import pandas as pd
import pytest

from src.reliability import deactivation as DA
from src.reliability import severity as SV


# ------------------------------------------------------- eq 61
def test_severity_is_zero_inside_limits_and_monotone_outside():
    t = SV.SeverityTerm("T_furnace", ref=400.0, scale=10.0, weight=1.0, gamma=2.0)
    assert t.contribution(380.0) == 0.0
    assert t.contribution(400.0) == 0.0
    v = np.array([405.0, 410.0, 420.0])
    c = t.contribution(v)
    assert np.all(np.diff(c) > 0)
    assert c[1] == pytest.approx(1.0)          # (410-400)/10 = 1, squared
    assert c[2] == pytest.approx(4.0)


def test_severity_direction_handles_a_lower_limit():
    t = SV.SeverityTerm("H2_pressure", ref=3.0, scale=0.5, direction=-1)
    assert t.contribution(3.5) == 0.0
    assert t.contribution(2.5) == pytest.approx(1.0)


def test_severity_index_sums_and_reports_the_dominant_proxy():
    idx = SV.SeverityIndex([
        SV.SeverityTerm("A", ref=0.0, scale=1.0, weight=1.0),
        SV.SeverityTerm("B", ref=0.0, scale=1.0, weight=3.0),
    ])
    out = idx.evaluate({"A": 2.0, "B": 1.0})
    assert out["S_sev"] == pytest.approx(4.0 + 3.0)
    assert out["dominant"] == "A"
    out2 = idx.evaluate({"A": 2.0})
    assert out2["missing"] == ["B"] and out2["S_sev"] == pytest.approx(4.0)


def test_severity_rejects_degenerate_scale():
    with pytest.raises(ValueError):
        SV.SeverityTerm("A", ref=0.0, scale=0.0).contribution(1.0)


def test_robust_reference_resists_injected_outliers():
    rng = np.random.default_rng(0)
    clean = rng.normal(400.0, 5.0, 2000)
    dirty = clean.copy()
    dirty[:20] = 1e6                              # frozen / spiked samples
    r1, s1 = SV.robust_reference(clean)
    r2, s2 = SV.robust_reference(dirty)
    assert abs(s2 - s1) / s1 < 0.05               # MAD barely moves
    assert abs(r2 - r1) < 1.0
    # a mean/std reference would move by orders of magnitude
    assert abs(dirty.mean() - clean.mean()) > 1000.0
    with pytest.raises(ValueError):
        SV.robust_reference([1.0, 2.0])


def test_robust_reference_survives_a_constant_series():
    ref, scale = SV.robust_reference(np.full(50, 7.0))
    assert scale > 0 and ref == pytest.approx(7.0)


def test_severity_frame_evaluation_ignores_missing_columns():
    idx = SV.SeverityIndex([SV.SeverityTerm("A", 0.0, 1.0),
                            SV.SeverityTerm("missing", 0.0, 1.0)])
    df = pd.DataFrame({"A": [0.0, 1.0, np.nan]})
    s = idx.evaluate_frame(df)
    assert list(s[:2]) == [0.0, 1.0] and s.iloc[2] == 0.0


def test_thermal_feedback_term_only_fires_when_unstable():
    assert SV.thermal_feedback_term(-1e-3) == 0.0
    assert SV.thermal_feedback_term(2e-4, scale=1e-4) == pytest.approx(4.0)


# ------------------------------------------------------- eq 62
def test_deactivation_detects_injected_drift_at_fixed_quality():
    rng = np.random.default_rng(1)
    n = 2000
    t_h = np.arange(n) * 1.0
    S = rng.uniform(5.0, 9.0, n)
    lhsv = rng.uniform(1.2, 1.8, n)
    pH2 = rng.uniform(2.5, 3.0, n)
    drift = 4.0 * t_h / 1000.0                    # +4 degC per 1000 h
    T = 350.0 + 2.0 * S - 3.0 * lhsv + 1.5 * pH2 + drift + rng.normal(0, 0.3, n)
    X = np.column_stack([S, lhsv, pH2])
    model = DA.fit_normalisation(X, T, ["S", "LHSV", "pH2"])
    dT = DA.delta_T_norm(T, X, model)
    rep = DA.drift_trend(t_h, dT)
    assert rep.significant
    assert rep.slope_c_per_1000h == pytest.approx(4.0, rel=0.2)
    assert "hypothesis" in rep.status.lower()
    assert "not a measured" in rep.note.lower()


def test_no_drift_is_not_flagged():
    rng = np.random.default_rng(2)
    n = 1500
    t_h = np.arange(n) * 1.0
    S = rng.uniform(5.0, 9.0, n)
    X = np.column_stack([S])
    T = 350.0 + 2.0 * S + rng.normal(0, 0.5, n)
    model = DA.fit_normalisation(X, T, ["S"])
    rep = DA.drift_trend(t_h, DA.delta_T_norm(T, X, model))
    assert not rep.significant


def test_bound_tightening_is_capped_and_explained():
    rep = DA.DeactivationReport(slope_c_per_1000h=12.0, p_value=1e-6, n=1000,
                                span_h=2000.0, significant=True)
    hi, why = DA.tighten_temperature_bound(375.0, rep, max_tighten_c=5.0)
    assert hi == pytest.approx(370.0)
    assert "[DOP]" in why and "deactivation hypothesis" in why
    quiet = DA.DeactivationReport(0.1, 0.9, 100, 500.0, False)
    hi2, why2 = DA.tighten_temperature_bound(375.0, quiet)
    assert hi2 == 375.0 and "no significant" in why2


def test_deactivation_guards():
    with pytest.raises(ValueError):
        DA.drift_trend([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        DA.fit_normalisation(np.zeros((3, 2)), np.zeros(3), ["a", "b"])
