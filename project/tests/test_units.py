"""Isolation checks for src/data/units.py  --  model_spec.md §1, eq (1)-(2).

Target: unit conversions.  Every conversion must round-trip and reproduce a
hand-computed value.
"""
import numpy as np
import pytest

from src.data import units as U


def test_temperature_roundtrip_and_reference():
    # closed-form: 0 degC == 273.15 K, 25 degC == 298.15 K
    assert U.c_to_k(0.0) == pytest.approx(273.15)
    assert U.c_to_k(25.0) == pytest.approx(298.15)
    x = np.array([-40.0, 0.0, 15.0, 360.0])
    assert np.allclose(U.k_to_c(U.c_to_k(x)), x, atol=1e-12)


def test_mass_flow_closed_form_and_inverse():
    # rho = 830 kg/m3, q = 0.1 m3/s -> 83 kg/s
    assert U.mass_flow(830.0, 0.1) == pytest.approx(83.0)
    assert U.volume_flow(830.0, 83.0) == pytest.approx(0.1)


def test_sulfur_fraction_roundtrip():
    # 10 mg/kg == 1e-5 kg/kg
    assert U.s_mgkg_to_wfrac(10.0) == pytest.approx(1e-5)
    s = np.array([0.5, 10.0, 1000.0])
    assert np.allclose(U.wfrac_to_s_mgkg(U.s_mgkg_to_wfrac(s)), s)
    # 0.2 %mass == 2000 mg/kg
    assert U.s_wtpct_to_mgkg(0.2) == pytest.approx(2000.0)
    assert U.s_mgkg_to_wtpct(2000.0) == pytest.approx(0.2)


def test_molar_concentration_eq2_hand_value():
    # eq 2: C_S = 1e-6 * rho * S / M_S
    # rho = 830, S = 10 mg/kg  ->  1e-6*830*10/0.03206 = 0.2589... mol/m3
    got = U.sulfur_molar_conc(830.0, 10.0)
    expected = 1e-6 * 830.0 * 10.0 / U.M_S
    assert got == pytest.approx(expected)
    assert got == pytest.approx(0.25889, rel=1e-4)
    # inverse
    assert U.sulfur_from_molar_conc(830.0, got) == pytest.approx(10.0)


def test_flow_and_pressure_roundtrips():
    assert U.tph_to_kg_s(3.6) == pytest.approx(1.0)
    assert U.kg_s_to_tph(1.0) == pytest.approx(3.6)
    assert U.m3h_to_m3_s(3600.0) == pytest.approx(1.0)
    assert U.m3_s_to_m3h(1.0) == pytest.approx(3600.0)
    assert U.mpa_to_pa(3.5) == pytest.approx(3.5e6)
    assert U.pa_to_mpa(3.5e6) == pytest.approx(3.5)
    assert U.kgfcm2_to_pa(1.0) == pytest.approx(98066.5)


def test_partial_pressure_eq17():
    # 80 mol% H2 at 3.5 MPa abs -> 2.8 MPa
    assert U.partial_pressure_h2(0.8, 3.5e6) == pytest.approx(2.8e6)


def test_units_are_vectorised_and_nan_safe():
    x = np.array([1.0, np.nan])
    out = U.c_to_k(x)
    assert np.isnan(out[1]) and out[0] == pytest.approx(274.15)
    assert U.c_to_k(np.array([])).shape == (0,)
