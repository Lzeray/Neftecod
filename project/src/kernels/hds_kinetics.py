"""HDS kinetics -- model_spec.md §6-7, eq (13), (16)-(18).

    (13)  r_HDS = k_HDS C_S^s (C_H2^s)^0.45 / (1 + K_H2S C_H2S^s)^2      [LIT]
    (16)  C_S^out = [ (C_S^in)^{1-n} + (n-1) k0 e^{-Ea/(R T)} p_H2^a / LHSV ]^{1/(1-n)}
    (17)  p_H2 = chi_H2 * p_abs
    (18)  C_H2^{L,*} = H p_H2      (equilibrium; the actual value is lower)

Equation 16 is a *reduced* plug-flow form [DOP] and is used only as a steady
state sanity check, never inside the dynamic model -- exactly as §6 requires.
"""
from __future__ import annotations

import numpy as np

from src.data.units import R_GAS
from src.kernels.arrhenius import k_arrhenius

#: exponent on the hydrogen surface concentration in eq 13 [LIT]
H2_ORDER = 0.45


def r_hds(C_S, C_H2, C_H2S, k_hds: float, K_H2S: float,
          h2_order: float = H2_ORDER):
    """Langmuir-Hinshelwood HDS rate, eq (13).

    Inputs are surface concentrations [mol/m^3]; the result carries the units
    of ``k_hds``.  Monotone: increasing in ``C_S`` and ``C_H2``, decreasing in
    ``C_H2S`` (H2S inhibition).
    """
    C_S = np.asarray(C_S, dtype=float)
    C_H2 = np.asarray(C_H2, dtype=float)
    C_H2S = np.asarray(C_H2S, dtype=float)
    if np.any(C_S < 0) or np.any(C_H2 < 0) or np.any(C_H2S < 0):
        raise ValueError("concentrations must be non-negative")
    return k_hds * C_S * np.power(C_H2, h2_order) / (1.0 + K_H2S * C_H2S) ** 2


def r_hds_T(C_S, C_H2, C_H2S, T_K, k0: float, Ea: float, K_H2S: float,
            h2_order: float = H2_ORDER, R: float = R_GAS):
    """eq (13) with the rate constant supplied by eq (14)."""
    return r_hds(C_S, C_H2, C_H2S, k_arrhenius(T_K, k0, Ea, R), K_H2S, h2_order)


def dr_dCS(C_S, C_H2, C_H2S, k_hds: float, K_H2S: float,
           h2_order: float = H2_ORDER):
    """Analytic d r/d C_S (first order in C_S) -- feeds eq 22."""
    C_H2 = np.asarray(C_H2, dtype=float)
    C_H2S = np.asarray(C_H2S, dtype=float)
    return k_hds * np.power(C_H2, h2_order) / (1.0 + K_H2S * C_H2S) ** 2


def dr_dT(C_S, C_H2, C_H2S, T_K, k0: float, Ea: float, K_H2S: float,
          h2_order: float = H2_ORDER, R: float = R_GAS):
    """Analytic d r/d T: only k(T) carries the temperature (eq 14-15)."""
    k = k_arrhenius(T_K, k0, Ea, R)
    base = r_hds(C_S, C_H2, C_H2S, 1.0, K_H2S, h2_order)
    return base * k * Ea / (R * np.asarray(T_K, dtype=float) ** 2)


def dr_dpH2(C_S, C_H2, C_H2S, k_hds: float, K_H2S: float, henry: float,
            h2_order: float = H2_ORDER):
    """d r/d p_H2 through eq 18 (C_H2 = henry * p_H2), chain rule."""
    C_H2 = np.asarray(C_H2, dtype=float)
    dC = henry
    with np.errstate(divide="ignore", invalid="ignore"):
        drdC = (k_hds * np.asarray(C_S, float) * h2_order
                * np.power(C_H2, h2_order - 1.0)
                / (1.0 + K_H2S * np.asarray(C_H2S, float)) ** 2)
    return drdC * dC


def h2_equilibrium_conc(p_H2_pa, henry_cp: float):
    """C_H2* = H^cp p_H2   (eq 18).  ``henry_cp`` in mol/(m^3 Pa)."""
    return henry_cp * np.asarray(p_H2_pa, dtype=float)


# ----------------------------------------------------------------------
def cs_out_plugflow(C_S_in, T_K, p_H2_pa, lhsv_h, k0: float, Ea: float,
                    n: float = 1.5, a: float = 1.0, R: float = R_GAS):
    """Reduced power-law plug-flow outlet concentration, eq (16)  [DOP].

    Steady-state sanity check only (model_spec.md §6).  ``n`` must be in
    (1, 2) for the closed form to be defined; ``n in [1.2, 1.7]`` per eq 16.
    """
    if not (1.0 < n < 2.0):
        raise ValueError("eq 16 requires 1 < n < 2")
    C_in = np.asarray(C_S_in, dtype=float)
    if np.any(C_in <= 0):
        raise ValueError("inlet concentration must be positive")
    lhsv = np.asarray(lhsv_h, dtype=float)
    if np.any(lhsv <= 0):
        raise ValueError("LHSV must be positive")
    k = k0 * np.exp(-Ea / (R * np.asarray(T_K, dtype=float)))
    term = np.power(C_in, 1.0 - n) + (n - 1.0) * k * np.power(
        np.asarray(p_H2_pa, dtype=float), a) / lhsv
    return np.power(term, 1.0 / (1.0 - n))


def wabt(T_in_C, T_out_C, weights: tuple[float, float] = (1 / 3, 2 / 3)):
    """Weighted-average bed temperature proxy, degC.

    [DOP] Only the inlet and outlet of the reactor train are instrumented, so
    the classical 1/3-2/3 WABT weighting is used and declared as an assumption.
    """
    w_in, w_out = weights
    if abs(w_in + w_out - 1.0) > 1e-12:
        raise ValueError("WABT weights must sum to 1")
    return w_in * np.asarray(T_in_C, float) + w_out * np.asarray(T_out_C, float)
