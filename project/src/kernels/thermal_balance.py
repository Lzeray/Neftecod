"""Reduced thermal balance -- model_spec.md §9, eq (24)-(25).

    (24)  C_th dT/dt = m_dot cp (T_in - T) + Q_heat - UA (T - T_env)
                       + (-dH) W_cat r_eff
    (25)  a_TT = [-m_dot0 cp - UA + (-dH) W_cat r_T0] / C_th
          a_TC = (-dH) W_cat r_C0 / C_th
          b_TQ = 1/C_th
          b_T,Tin = m_dot0 cp / C_th

Units: C_th [J/K], m_dot [kg/s], cp [J/(kg K)], Q [W], UA [W/K],
dH [J/mol], W_cat [kg], r_eff [mol/(kg s)], T [K].

model_spec.md §9: the exotherm can make ``a_TT`` positive -- a runaway mode.
``ThermalLinearisation.is_stable`` exposes that explicitly and it feeds the
severity index (§29).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalParams:
    """[DOP] lumped thermal parameters -- see assumptions register."""
    C_th_J_K: float
    cp_J_kgK: float
    UA_W_K: float
    dH_J_mol: float          # negative for an exothermic reaction
    W_cat_kg: float


def dT_dt(T_K, T_in_K, m_dot_kg_s, Q_heat_W, T_env_K, r_eff, p: ThermalParams):
    """Right-hand side of eq (24), [K/s]."""
    T = np.asarray(T_K, float)
    return (np.asarray(m_dot_kg_s, float) * p.cp_J_kgK
            * (np.asarray(T_in_K, float) - T)
            + np.asarray(Q_heat_W, float)
            - p.UA_W_K * (T - np.asarray(T_env_K, float))
            + (-p.dH_J_mol) * p.W_cat_kg * np.asarray(r_eff, float)) / p.C_th_J_K


@dataclass
class ThermalLinearisation:
    a_TT: float
    a_TC: float
    b_TQ: float
    b_T_Tin: float

    @property
    def is_stable(self) -> bool:
        """a_TT < 0.  A positive a_TT is the exothermic runaway mode of §9."""
        return self.a_TT < 0

    @property
    def tau_T_s(self) -> float:
        return float(-1.0 / self.a_TT) if self.a_TT < 0 else float("inf")


def linearise_thermal(p: ThermalParams, m_dot0_kg_s: float, r_T0: float,
                      r_C0: float) -> ThermalLinearisation:
    """Coefficients of eq (25)."""
    a_TT = (-m_dot0_kg_s * p.cp_J_kgK - p.UA_W_K
            + (-p.dH_J_mol) * p.W_cat_kg * r_T0) / p.C_th_J_K
    a_TC = (-p.dH_J_mol) * p.W_cat_kg * r_C0 / p.C_th_J_K
    return ThermalLinearisation(a_TT, a_TC, 1.0 / p.C_th_J_K,
                                m_dot0_kg_s * p.cp_J_kgK / p.C_th_J_K)


def coupled_A(sulf, therm) -> np.ndarray:
    """2x2 continuous state matrix of the coupled (C_S, T) modes, §8-§9.

    State order: ``[dC_S, dT]``.  The off-diagonal terms are what makes the
    sulfur and temperature dynamics inseparable (model_spec.md §9).
    """
    return np.array([[sulf.a_CC, sulf.a_CT],
                     [therm.a_TC, therm.a_TT]], dtype=float)
