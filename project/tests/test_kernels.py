"""Isolation checks for Block C -- model_spec.md §6-9, §21 (eq 13-25, 46).

Targets and how each is verified independently of the pipeline:
  * arrhenius (eq 14-15) : eq 15 vs a finite difference of eq 14;
                           fit_arrhenius recovers planted (k0, Ea)
  * hds_kinetics (eq 13) : sign/monotonicity in C_S, C_H2, C_H2S, T;
                           analytic partials vs finite differences;
                           eq 16 reproduces eq 13 in the low-inhibition limit
  * reactor_balance (22) : coefficients vs the numeric Jacobian of eq 19
  * thermal_balance (25) : coefficients vs the numeric Jacobian of eq 24
"""
import numpy as np
import pytest

from src.data.units import R_GAS, c_to_k
from src.kernels import arrhenius as AR
from src.kernels import hds_kinetics as HK
from src.kernels import reactor_balance as RB
from src.kernels import thermal_balance as TB

# a representative HDS operating point
T0 = c_to_k(350.0)
K0, EA = 1.2e6, 90_000.0
KH2S = 1.5
CS0, CH20, CH2S0 = 8.0, 300.0, 2.0


# ------------------------------------------------------- eq 14-15
def test_arrhenius_two_forms_agree():
    T = np.linspace(600.0, 680.0, 7)
    kT0 = AR.k_arrhenius(T0, K0, EA)
    assert np.allclose(AR.k_arrhenius(T, K0, EA),
                       AR.k_ref_form(T, kT0, T0, EA), rtol=1e-12)


def test_eq15_matches_finite_difference_of_eq14():
    kT0 = float(AR.k_arrhenius(T0, K0, EA))
    analytic = AR.dk_dT(T0, kT0, EA)
    h = 1e-4
    numeric = (AR.k_arrhenius(T0 + h, K0, EA)
               - AR.k_arrhenius(T0 - h, K0, EA)) / (2 * h)
    assert analytic == pytest.approx(float(numeric), rel=1e-8)


def test_arrhenius_rejects_celsius():
    with pytest.raises(ValueError):
        AR.k_arrhenius(-10.0, K0, EA)          # a Celsius value would be < 0 K


def test_fit_arrhenius_recovers_parameters():
    T = np.linspace(600.0, 700.0, 20)
    k = AR.k_arrhenius(T, K0, EA)
    k0_hat, ea_hat = AR.fit_arrhenius(T, k)
    assert k0_hat == pytest.approx(K0, rel=1e-6)
    assert ea_hat == pytest.approx(EA, rel=1e-6)
    with pytest.raises(ValueError):
        AR.fit_arrhenius([600.0], [1.0])


# ------------------------------------------------------- eq 13
def test_lh_rate_signs_and_monotonicity():
    base = HK.r_hds(CS0, CH20, CH2S0, 1.0, KH2S)
    assert base > 0
    assert HK.r_hds(CS0 * 1.1, CH20, CH2S0, 1.0, KH2S) > base      # more S
    assert HK.r_hds(CS0, CH20 * 1.1, CH2S0, 1.0, KH2S) > base      # more H2
    assert HK.r_hds(CS0, CH20, CH2S0 * 1.1, 1.0, KH2S) < base      # H2S inhibits
    # temperature
    r1 = HK.r_hds_T(CS0, CH20, CH2S0, T0, K0, EA, KH2S)
    r2 = HK.r_hds_T(CS0, CH20, CH2S0, T0 + 5.0, K0, EA, KH2S)
    assert r2 > r1


def test_lh_rate_edge_cases():
    assert HK.r_hds(0.0, CH20, CH2S0, 1.0, KH2S) == 0.0
    with pytest.raises(ValueError):
        HK.r_hds(-1.0, CH20, CH2S0, 1.0, KH2S)
    # no inhibition -> denominator is exactly 1
    assert HK.r_hds(CS0, CH20, 0.0, 1.0, KH2S) == pytest.approx(
        CS0 * CH20 ** HK.H2_ORDER)


def test_analytic_partials_match_finite_differences():
    k = float(AR.k_arrhenius(T0, K0, EA))
    h = 1e-6
    # d r / d C_S
    num = (HK.r_hds(CS0 + h, CH20, CH2S0, k, KH2S)
           - HK.r_hds(CS0 - h, CH20, CH2S0, k, KH2S)) / (2 * h)
    assert float(HK.dr_dCS(CS0, CH20, CH2S0, k, KH2S)) == pytest.approx(
        float(num), rel=1e-6)
    # d r / d T
    hT = 1e-3
    numT = (HK.r_hds_T(CS0, CH20, CH2S0, T0 + hT, K0, EA, KH2S)
            - HK.r_hds_T(CS0, CH20, CH2S0, T0 - hT, K0, EA, KH2S)) / (2 * hT)
    assert float(HK.dr_dT(CS0, CH20, CH2S0, T0, K0, EA, KH2S)) == pytest.approx(
        float(numT), rel=1e-5)
    # d r / d p_H2 through eq 18
    henry = 2.5e-5
    p0 = 2.8e6
    hp = 1.0
    def r_of_p(p):
        return HK.r_hds(CS0, HK.h2_equilibrium_conc(p, henry), CH2S0, k, KH2S)
    nump = (r_of_p(p0 + hp) - r_of_p(p0 - hp)) / (2 * hp)
    got = HK.dr_dpH2(CS0, HK.h2_equilibrium_conc(p0, henry), CH2S0, k, KH2S,
                     henry)
    assert float(got) == pytest.approx(float(nump), rel=1e-5)


def test_eq16_reduced_form_matches_eq13_in_low_inhibition_limit():
    """eq 16 is a plug-flow first-order-in-S closed form.  With n -> 1+ and no
    inhibition it must agree with integrating the LH rate along the reactor."""
    n = 1.0 + 1e-6
    C_in, T, pH2, lhsv = 100.0, T0, 2.8e6, 1.5
    a = 0.0                                    # drop the p_H2 exponent
    out = HK.cs_out_plugflow(C_in, T, pH2, lhsv, K0, EA, n=n, a=a)
    # analytic first-order plug flow: C_out = C_in exp(-k/LHSV)
    k = K0 * np.exp(-EA / (R_GAS * T))
    assert float(out) == pytest.approx(C_in * np.exp(-k / lhsv), rel=1e-4)
    # and it is monotone: hotter and slower -> deeper desulfurisation
    hot = HK.cs_out_plugflow(C_in, T + 10, pH2, lhsv, K0, EA, n=1.5)
    ref = HK.cs_out_plugflow(C_in, T, pH2, lhsv, K0, EA, n=1.5)
    slow = HK.cs_out_plugflow(C_in, T, pH2, lhsv * 0.8, K0, EA, n=1.5)
    assert hot < ref and slow < ref


def test_eq16_guards():
    with pytest.raises(ValueError):
        HK.cs_out_plugflow(100.0, T0, 2.8e6, 1.5, K0, EA, n=1.0)
    with pytest.raises(ValueError):
        HK.cs_out_plugflow(100.0, T0, 2.8e6, 0.0, K0, EA, n=1.5)


def test_wabt_weights():
    assert HK.wabt(340.0, 370.0) == pytest.approx(340 / 3 + 2 * 370 / 3)
    with pytest.raises(ValueError):
        HK.wabt(340.0, 370.0, (0.5, 0.6))


# ------------------------------------------------------- eq 19-23
GEOM = RB.ReactorGeometry(V_L_m3=40.0, W_cat_kg=90_000.0)


def _rate(C_S, T_K):
    return HK.r_hds_T(C_S, CH20, CH2S0, T_K, K0, EA, KH2S)


def test_eq22_coefficients_match_numeric_jacobian_of_eq19():
    q0 = 0.05
    C_in0 = 300.0
    # find the consistent steady state (eq 20)
    C0 = RB.solve_steady_CS(C_in0, q0, GEOM, lambda c: _rate(c, T0))
    assert RB.steady_state_residual(C0, C_in0, q0, _rate(C0, T0), GEOM) \
        == pytest.approx(0.0, abs=1e-8)

    k = float(AR.k_arrhenius(T0, K0, EA))
    lin = RB.linearise_sulfur(
        GEOM, q0, C_in0, C0,
        dr_dCS0=float(HK.dr_dCS(C0, CH20, CH2S0, k, KH2S)),
        dr_dT0=float(HK.dr_dT(C0, CH20, CH2S0, T0, K0, EA, KH2S)),
    )

    # numeric Jacobian of eq 19 w.r.t. (C_S, T, q, C_S,in)
    def f(x):
        C, T, q, Cin = x
        return RB.dCS_dt(C, Cin, q, _rate(C, T), GEOM)

    J = RB.numeric_jacobian(f, np.array([C0, T0, q0, C_in0]))
    assert lin.a_CC == pytest.approx(J[0, 0], rel=1e-5)
    assert lin.a_CT == pytest.approx(J[0, 1], rel=1e-4)
    assert lin.b_Cq == pytest.approx(J[0, 2], rel=1e-5)
    assert lin.e_Cin == pytest.approx(J[0, 3], rel=1e-6)


def test_tau_C_eq23_and_stability():
    lin = RB.linearise_sulfur(GEOM, 0.05, 300.0, 8.0, dr_dCS0=1e-5, dr_dT0=0.0)
    assert lin.a_CC < 0
    assert lin.tau_C_s == pytest.approx(-1.0 / lin.a_CC)
    # tau_C is NOT simply V_L/q0 -- the reaction shortens it (§8)
    assert lin.tau_C_s < GEOM.V_L_m3 / 0.05
    unstable = RB.SulfurLinearisation(0.1, 0, 0, 0, 0)
    assert np.isinf(unstable.tau_C_s)


def test_solve_steady_rejects_inconsistent_point():
    with pytest.raises(ValueError):
        RB.solve_steady_CS(300.0, 0.05, GEOM, lambda c: -1.0)


# ------------------------------------------------------- eq 24-25
TP = TB.ThermalParams(C_th_J_K=4.0e7, cp_J_kgK=2200.0, UA_W_K=3000.0,
                      dH_J_mol=-250_000.0, W_cat_kg=90_000.0)


def test_eq25_coefficients_match_numeric_jacobian_of_eq24():
    m0, Tin0, Q0, Tenv = 40.0, T0 - 20.0, 0.0, c_to_k(20.0)
    C0 = 8.0
    r_T0 = float(HK.dr_dT(C0, CH20, CH2S0, T0, K0, EA, KH2S))
    r_C0 = float(HK.dr_dCS(C0, CH20, CH2S0,
                           float(AR.k_arrhenius(T0, K0, EA)), KH2S))
    lin = TB.linearise_thermal(TP, m0, r_T0, r_C0)

    def f(x):
        T, C, Q, Tin = x
        return TB.dT_dt(T, Tin, m0, Q, Tenv, _rate(C, T), TP)

    # Q_heat sits at 0 W at this operating point, so it needs its own step
    # scale (1 MW) for the finite difference to carry any significant digits.
    J = RB.numeric_jacobian(f, np.array([T0, C0, Q0, Tin0]),
                            scale=np.array([1.0, 1.0, 1e6, 1.0]))
    assert lin.a_TT == pytest.approx(J[0, 0], rel=1e-5)
    assert lin.a_TC == pytest.approx(J[0, 1], rel=1e-5)
    assert lin.b_TQ == pytest.approx(J[0, 2], rel=1e-6)
    assert lin.b_T_Tin == pytest.approx(J[0, 3], rel=1e-6)


def test_exothermic_runaway_is_detected_not_hidden():
    """A large exotherm can flip a_TT positive -- §9 risk factor."""
    hot = TB.ThermalParams(C_th_J_K=1.0e5, cp_J_kgK=2200.0, UA_W_K=10.0,
                           dH_J_mol=-2.5e6, W_cat_kg=9.0e5)
    r_T0 = float(HK.dr_dT(CS0, CH20, CH2S0, T0, K0, EA, KH2S))
    lin = TB.linearise_thermal(hot, 0.01, r_T0, 0.0)
    assert lin.a_TT > 0 and not lin.is_stable and np.isinf(lin.tau_T_s)


def test_coupled_A_is_not_diagonal():
    slin = RB.linearise_sulfur(GEOM, 0.05, 300.0, 8.0, 1e-5, -1e-7)
    tlin = TB.linearise_thermal(TP, 40.0, 1e-7, 1e-5)
    A = TB.coupled_A(slin, tlin)
    assert A.shape == (2, 2)
    assert A[0, 1] != 0.0 and A[1, 0] != 0.0
    # continuous-time stability check (eq 45)
    assert np.all(np.real(np.linalg.eigvals(A)) < 0)
