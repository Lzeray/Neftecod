"""Arrhenius temperature dependence -- model_spec.md §6, eq (14)-(15).

    (14)  k(T) = k0 exp(-Ea/(R T)) = k(T0) exp[-(Ea/R)(1/T - 1/T0)]
    (15)  dk/dT|_0 = k(T0) * Ea / (R T0^2)

Units: T [K], Ea [J/mol], R [J/(mol K)], k in whatever units k0 carries.

model_spec.md §6 warning: a rising ``k`` does **not** prove the sign of the
total temperature effect -- adsorption, mass transfer and side reactions move
at the same time.  Nothing downstream may use eq 15 as "the" temperature gain;
it is only the kinetic contribution.
"""
from __future__ import annotations

import numpy as np

from src.data.units import R_GAS


def k_arrhenius(T_K, k0: float, Ea: float, R: float = R_GAS):
    """k(T) = k0 exp(-Ea/(R T))   (eq 14, absolute form)."""
    T = np.asarray(T_K, dtype=float)
    if np.any(T <= 0):
        raise ValueError("absolute temperature must be positive (K, not degC)")
    return k0 * np.exp(-Ea / (R * T))


def k_ref_form(T_K, k_T0: float, T0_K: float, Ea: float, R: float = R_GAS):
    """k(T) = k(T0) exp[-(Ea/R)(1/T - 1/T0)]   (eq 14, reference form)."""
    T = np.asarray(T_K, dtype=float)
    if np.any(T <= 0) or T0_K <= 0:
        raise ValueError("absolute temperature must be positive (K, not degC)")
    return k_T0 * np.exp(-(Ea / R) * (1.0 / T - 1.0 / T0_K))


def dk_dT(T0_K: float, k_T0: float, Ea: float, R: float = R_GAS) -> float:
    """dk/dT at T0   (eq 15)."""
    if T0_K <= 0:
        raise ValueError("absolute temperature must be positive (K, not degC)")
    return float(k_T0 * Ea / (R * T0_K ** 2))


def fit_arrhenius(T_K, k_obs, R: float = R_GAS) -> tuple[float, float]:
    """Least squares on ln k = ln k0 - Ea/(R T);  returns ``(k0, Ea)``.

    Used only as an initialiser for the non-linear kinetic fit (§37 step 2).
    """
    T = np.asarray(T_K, dtype=float)
    k = np.asarray(k_obs, dtype=float)
    m = np.isfinite(T) & np.isfinite(k) & (k > 0) & (T > 0)
    if m.sum() < 2:
        raise ValueError("need >= 2 valid (T, k) points")
    A = np.column_stack([np.ones(m.sum()), -1.0 / (R * T[m])])
    coef, *_ = np.linalg.lstsq(A, np.log(k[m]), rcond=None)
    return float(np.exp(coef[0])), float(coef[1])
