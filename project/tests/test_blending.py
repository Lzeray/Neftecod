"""Isolation checks for Block H -- model_spec.md §4, §21 (eq 6-9, 46).

Targets:
  * blending.blend_sulfur_mass / blend_linear -- eq 6-7, linear in mass
  * blending.delta_sulfur                     -- eq 8 vs a finite difference of eq 6
  * blending.simulate_mixer                   -- eq 9, tau_mix and steady state
  * indices.blend_indexed                     -- eq 46, and the refusal to blend
    a non-linear property linearly
"""
import numpy as np
import pytest

from src.blend import blending as B
from src.blend import indices as IX


# ------------------------------------------------------- eq 6-7
def test_two_component_sulfur_hand_example():
    # 70 kg/s at 2 mg/kg + 30 kg/s at 9 mg/kg -> (140+270)/100 = 4.1
    assert B.blend_sulfur_mass([70.0, 30.0], [2.0, 9.0]) == pytest.approx(4.1)
    assert B.blend_linear([0.7, 0.3], [2.0, 9.0]) == pytest.approx(4.1)


def test_sulfur_blend_is_exactly_linear_in_mass():
    rng = np.random.default_rng(0)
    S = rng.uniform(1, 20, 4)
    w1, w2 = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    lam = 0.37
    mixed = lam * w1 + (1 - lam) * w2
    assert B.blend_linear(mixed, S) == pytest.approx(
        lam * B.blend_linear(w1, S) + (1 - lam) * B.blend_linear(w2, S))


def test_simplex_enforced_and_renormalised():
    with pytest.raises(ValueError):
        B.blend_linear([0.5, 0.4], [1.0, 2.0])          # does not sum to 1
    with pytest.raises(ValueError):
        B.check_simplex([-0.1, 1.1])
    w = B.normalise_fractions([2.0, -1.0, 2.0])
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert list(w) == [0.5, 0.0, 0.5]
    with pytest.raises(ValueError):
        B.normalise_fractions([0.0, 0.0])


def test_blend_sulfur_mass_guards():
    with pytest.raises(ValueError):
        B.blend_sulfur_mass([0.0, 0.0], [1.0, 2.0])


# ------------------------------------------------------- eq 8
def test_delta_form_matches_finite_difference_of_eq6():
    m0 = np.array([70.0, 30.0])
    S0 = np.array([2.0, 9.0])
    S_mix0 = B.blend_sulfur_mass(m0, S0)
    p0 = m0 / m0.sum()
    dm0 = np.array([0.5, -0.2])
    dS0 = np.array([0.01, -0.03])
    # eq 8 is the first-order term of eq 6, so the relative error of the linear
    # form must shrink like O(eps) as the perturbation shrinks.
    errs = []
    for eps in (1.0, 0.1, 0.01):
        lin = B.delta_sulfur(p0, S0, S_mix0, m0.sum(), eps * dS0, eps * dm0)
        exact = B.blend_sulfur_mass(m0 + eps * dm0, S0 + eps * dS0) - S_mix0
        errs.append(abs(lin - exact) / abs(exact))
    assert errs[0] < 1e-2
    assert errs[1] < 0.15 * errs[0] and errs[2] < 0.15 * errs[1]


# ------------------------------------------------------- eq 9
def test_mixer_converges_to_eq6_steady_state_with_tau_mix():
    m = np.array([70.0, 30.0])          # kg/s
    S_in = np.array([2.0, 9.0])
    M = 10_000.0                        # kg
    tau = B.tau_mix(M, m.sum())
    assert tau == pytest.approx(100.0)
    traj = B.simulate_mixer(0.0, m, S_in, M, dt_s=1.0, n_steps=2000)
    assert traj[-1] == pytest.approx(B.blend_sulfur_mass(m, S_in), rel=1e-6)
    # after one tau, ~63% of the way (first-order response)
    assert traj[100] == pytest.approx(0.632 * traj[-1], rel=2e-2)
    with pytest.raises(ValueError):
        B.tau_mix(M, 0.0)


# ------------------------------------------------------- eq 46
def test_index_roundtrips():
    for name, idx in IX.INDEXED_PROPERTIES.items():
        y = np.array([-20.0, -5.0, 5.0])          # cold-flow temperatures, degC
        if name == "T95":
            y = np.array([340.0, 360.0])          # degC
        elif name == "FlashPoint":
            y = np.array([55.0, 62.0, 80.0])      # degC
        elif name == "Viscosity":
            y = np.array([1.8, 3.0, 6.0])         # cSt -- strictly positive
        assert np.allclose(idx.inv(idx(y)), y, atol=1e-6), name


def test_indexed_blend_is_not_linear_and_is_bracketed():
    y = np.array([-25.0, -5.0])
    w = np.array([0.5, 0.5])
    got = B.blend_property("CFPP", w, y)
    lin = 0.5 * (-25.0) + 0.5 * (-5.0)
    assert min(y) <= got <= max(y)          # a blend cannot leave the range
    assert abs(got - lin) > 1e-6, "CFPP must NOT blend linearly (§4, §21)"


def test_flash_and_t95_use_the_index_path():
    for prop, y in (("FlashPoint", np.array([56.0, 80.0])),
                    ("T95", np.array([340.0, 365.0]))):
        w = np.array([0.5, 0.5])
        got = B.blend_property(prop, w, y)
        assert min(y) <= got <= max(y)
        assert got != pytest.approx(float(np.mean(y)), abs=1e-6)


def test_pure_component_blend_returns_the_component():
    y = np.array([-12.0, 3.0])
    assert B.blend_property("CFPP", np.array([1.0, 0.0]), y) == pytest.approx(-12.0)


def test_unknown_property_is_rejected_not_blended_linearly():
    with pytest.raises(KeyError):
        B.blend_property("Viscosity_at_100C", np.array([0.5, 0.5]),
                         np.array([2.0, 4.0]))


def test_linear_properties_use_eq7():
    assert B.blend_property("D15", np.array([0.25, 0.75]),
                            np.array([820.0, 840.0])) == pytest.approx(835.0)


def test_indexed_blend_requires_simplex():
    with pytest.raises(ValueError):
        IX.blend_indexed(np.array([0.5, 0.6]), np.array([-5.0, -10.0]),
                         IX.CFPP_INDEX)


# ------------------------------------------------------- depressant
def test_depressant_dose_response_saturates_and_is_monotone():
    d = np.array([0.0, 50.0, 200.0, 1000.0, 1e6])
    shift = IX.depressant_shift(d, dmax_c=8.0, k_ppm=150.0)
    assert shift[0] == 0.0
    assert np.all(np.diff(shift) < 0)          # monotone decreasing
    assert shift[-1] == pytest.approx(-8.0, abs=1e-2)   # saturates at -dmax
    assert np.all(shift >= -8.0)
    with pytest.raises(ValueError):
        IX.depressant_shift([-1.0], 8.0, 150.0)


def test_depressant_applied_through_blend_property():
    y = np.array([-5.0, -5.0])
    w = np.array([0.5, 0.5])
    base = B.blend_property("CFPP", w, y)
    dosed = B.blend_property("CFPP", w, y, dose_ppm=300.0,
                             depressant={"dmax_c": 8.0, "k_ppm": 150.0})
    assert dosed < base
