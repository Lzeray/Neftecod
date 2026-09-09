"""Reduced dynamic sulfur balance -- model_spec.md §8, eq (19)-(23).

    (19)  V_L dC_S/dt = q (C_S,in - C_S) - W_cat r_eff
    (20)  q0 (C_S,in,0 - C_S,0) = W_cat r_eff,0                      (steady state)
    (21)  dC_S/dt|_lin = a_CC dC_S + a_CT dT + b_Cq dq + e_Cin dC_S,in + e_Cp dp_H2
    (22)  a_CC = -q0/V_L - (W_cat/V_L) dr/dC_S
          a_CT = -(W_cat/V_L) dr/dT
          b_Cq = (C_S,in,0 - C_S,0)/V_L
          e_Cin = q0/V_L
          e_Cp  = -(W_cat/V_L) dr/dp_H2
    (23)  tau_C = -1/a_CC

Units: V_L [m^3], q [m^3/s], C_S [mol/m^3], W_cat [kg], r_eff [mol/(kg s)].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReactorGeometry:
    """Liquid hold-up and catalyst load.  [DOP] -- see assumptions register."""
    V_L_m3: float
    W_cat_kg: float


def dCS_dt(C_S, C_S_in, q_m3_s, r_eff, geom: ReactorGeometry):
    """Right-hand side of eq (19), [mol/(m^3 s)]."""
    return (np.asarray(q_m3_s, float) * (np.asarray(C_S_in, float)
                                         - np.asarray(C_S, float))
            - geom.W_cat_kg * np.asarray(r_eff, float)) / geom.V_L_m3


def steady_state_residual(C_S, C_S_in, q_m3_s, r_eff, geom: ReactorGeometry):
    """eq (20) residual; zero at a consistent operating point."""
    return (np.asarray(q_m3_s, float)
            * (np.asarray(C_S_in, float) - np.asarray(C_S, float))
            - geom.W_cat_kg * np.asarray(r_eff, float))


def solve_steady_CS(C_S_in: float, q_m3_s: float, geom: ReactorGeometry,
                    r_fun, lo: float = 1e-12, hi: float | None = None,
                    tol: float = 1e-14, max_iter: int = 200) -> float:
    """Solve eq (20) for C_S given a rate function ``r_fun(C_S) -> r_eff``.

    Bisection on the monotone residual q(C_in - C) - W r(C): decreasing in C
    for any rate that is non-decreasing in C, so the root is unique.
    """
    hi = float(C_S_in) if hi is None else float(hi)

    def g(c):
        return q_m3_s * (C_S_in - c) - geom.W_cat_kg * float(r_fun(c))

    a, b = lo, hi
    ga, gb = g(a), g(b)
    if ga * gb > 0:
        raise ValueError("no sign change: check the operating point")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        gm = g(m)
        if abs(gm) < tol or (b - a) < tol:
            return m
        if ga * gm <= 0:
            b, gb = m, gm
        else:
            a, ga = m, gm
    return 0.5 * (a + b)


# ----------------------------------------------------------------------
@dataclass
class SulfurLinearisation:
    """Coefficients of eq (21)-(22) at one operating point."""
    a_CC: float
    a_CT: float
    b_Cq: float
    e_Cin: float
    e_Cp: float

    @property
    def tau_C_s(self) -> float:
        """tau_C = -1/a_CC   (eq 23); +inf if the mode is not stable."""
        return float(-1.0 / self.a_CC) if self.a_CC < 0 else float("inf")


def linearise_sulfur(geom: ReactorGeometry, q0_m3_s: float, C_S_in0: float,
                     C_S0: float, dr_dCS0: float, dr_dT0: float,
                     dr_dpH2_0: float = 0.0,
                     dr_dq0: float = 0.0) -> SulfurLinearisation:
    """Coefficients of eq (22).

    ``dr_dq0`` is the optional mass-transfer term noted under eq 22; when it is
    non-zero the extra ``-(W_cat/V_L) dr/dq`` contribution is added to b_Cq.
    """
    V, W = geom.V_L_m3, geom.W_cat_kg
    a_CC = -q0_m3_s / V - (W / V) * dr_dCS0
    a_CT = -(W / V) * dr_dT0
    b_Cq = (C_S_in0 - C_S0) / V - (W / V) * dr_dq0
    e_Cin = q0_m3_s / V
    e_Cp = -(W / V) * dr_dpH2_0
    return SulfurLinearisation(a_CC, a_CT, b_Cq, e_Cin, e_Cp)


def numeric_jacobian(f, x0: np.ndarray, eps: float = 1e-6,
                     scale: np.ndarray | None = None) -> np.ndarray:
    """Central-difference Jacobian, used to verify eq 22 and eq 25.

    ``scale`` gives the characteristic magnitude of each coordinate.  Without
    it a coordinate whose operating value happens to be 0 (an unfired heater,
    Q_heat = 0 W) would be perturbed by 1e-6 W, which is lost in the rounding
    of a ~1e-3 K/s derivative; the check would then fail for numerical, not
    physical, reasons.
    """
    x0 = np.asarray(x0, dtype=float)
    sc = np.ones_like(x0) if scale is None else np.abs(np.asarray(scale, float))
    f0 = np.atleast_1d(np.asarray(f(x0), dtype=float))
    J = np.zeros((f0.size, x0.size))
    for j in range(x0.size):
        h = eps * max(sc[j], abs(x0[j]))
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.atleast_1d(np.asarray(f(xp), float))
                   - np.atleast_1d(np.asarray(f(xm), float))) / (2 * h)
    return J
