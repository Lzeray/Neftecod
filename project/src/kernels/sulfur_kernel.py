"""Physical kernel for product sulfur -- model_spec.md §6, eq (2), (14), (16).

The plant supplies neither catalyst mass nor liquid hold-up, so the absolute
kinetic constants of eq 13 are not identifiable.  What *is* identifiable from
the package is the reduced plug-flow form of eq 16 written in *relative* LHSV:

    C_out = [ C_in^{1-n} + (n-1) k(T) p_H2^a / LHSV_rel ]^{1/(1-n)}
    k(T)  = k_ref exp[-(Ea/R)(1/T - 1/T_ref)]                       (eq 14)

with ``LHSV_rel`` the feed rate normalised by its training-window median.  The
lumped constant ``k_ref`` therefore absorbs W_cat/V_L and the absolute LHSV;
the identifiable parameters are ``(k_ref, Ea, n, a)``.  This is exactly the
"reduced form as a steady-state anchor" that §6 sanctions, and it is used as
the kernel g_phi of eq 55, never as the whole model.

Concentrations go through eq 2 so the kinetics see mol/m^3, not mg/kg.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.data.units import (R_GAS, c_to_k, sulfur_from_molar_conc,
                            sulfur_molar_conc)


@dataclass
class SulfurKernelParams:
    k_ref: float
    Ea: float
    n: float
    a: float
    T_ref_K: float
    rho_kg_m3: float = 830.0        # [DOP] product density used in eq 2
    status: str = "[DOP/LIT]"
    fit_report: dict | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "fit_report"}
        d["fit_report"] = self.fit_report
        return d


def predict_sulfur(S_in_mgkg, T_C, pH2_MPa, lhsv_rel, p: SulfurKernelParams):
    """Product sulfur [mg/kg] from eq 16 (+ eq 2 conversions)."""
    S_in = np.asarray(S_in_mgkg, float)
    T = c_to_k(np.asarray(T_C, float))
    pH2 = np.asarray(pH2_MPa, float)
    lh = np.asarray(lhsv_rel, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        C_in = sulfur_molar_conc(p.rho_kg_m3, np.clip(S_in, 1e-6, None))
        k = p.k_ref * np.exp(-(p.Ea / R_GAS) * (1.0 / T - 1.0 / p.T_ref_K))
        term = (np.power(C_in, 1.0 - p.n)
                + (p.n - 1.0) * k * np.power(np.clip(pH2, 1e-6, None), p.a)
                / np.clip(lh, 1e-6, None))
        C_out = np.power(np.clip(term, 1e-30, None), 1.0 / (1.0 - p.n))
        out = sulfur_from_molar_conc(p.rho_kg_m3, C_out)
    return np.where(np.isfinite(out), out, np.nan)


#: [LIT] Apparent activation energy of deep HDS of straight-run gas oil,
#: 80-150 kJ/mol in the literature; 100 kJ/mol is taken as the central value.
#: It is *fixed*, not fitted: on closed-loop data the temperature and the
#: sulfur are confounded (operators raise T *because* S is high), so a free fit
#: drives Ea to its lower bound and destroys the kinetic sign that the whole
#: recommendation depends on.  Using a literature value where the data cannot
#: identify one is exactly what the [LIT]/[DOP] distinction is for.
EA_LITERATURE_J_MOL = 100_000.0


def fit_sulfur_kernel(S_in_mgkg, T_C, pH2_MPa, lhsv_rel, S_out_mgkg,
                      T_ref_C: float | None = None, rho: float = 830.0,
                      n0: float = 1.5, huber_delta: float = 2.0,
                      Ea_fixed: float | None = EA_LITERATURE_J_MOL
                      ) -> SulfurKernelParams:
    """Non-linear least squares fit of eq 16 on stable periods (§37 step 2).

    A soft-L1 loss is used so the handful of off-spec laboratory results
    (the record contains values above 100 mg/kg) cannot drag the kinetics --
    §17 applied to the kernel fit.  ``n`` is constrained to (1, 2) as eq 16
    requires.  ``Ea_fixed`` pins the activation energy at a literature value;
    pass ``None`` to fit it and see, in the report, that it collapses to the
    bound.
    """
    S_in = np.asarray(S_in_mgkg, float)
    T = np.asarray(T_C, float)
    pH2 = np.asarray(pH2_MPa, float)
    lh = np.asarray(lhsv_rel, float)
    y = np.asarray(S_out_mgkg, float)
    m = (np.isfinite(S_in) & np.isfinite(T) & np.isfinite(pH2)
         & np.isfinite(lh) & np.isfinite(y) & (y > 0) & (S_in > 0))
    if m.sum() < 30:
        raise ValueError(f"only {m.sum()} usable rows for the kinetic fit")
    T_ref = float(np.median(T[m])) if T_ref_C is None else float(T_ref_C)
    T_ref_K = float(c_to_k(T_ref))

    def unpack(theta):
        if Ea_fixed is None:
            return theta[0], theta[1], theta[2], theta[3]
        return theta[0], float(Ea_fixed), theta[1], theta[2]

    def resid(theta):
        log_k, Ea, n, a = unpack(theta)
        p = SulfurKernelParams(np.exp(log_k), Ea, n, a, T_ref_K, rho)
        pred = predict_sulfur(S_in[m], T[m], pH2[m], lh[m], p)
        r = np.log(np.clip(pred, 1e-6, None)) - np.log(np.clip(y[m], 1e-6, None))
        return np.where(np.isfinite(r), r, 10.0)

    if Ea_fixed is None:
        x0s = [[lk, 80_000.0, n0, 1.0] for lk in (-6.0, -3.0, 0.0, 3.0)]
        lo, hi = ([-20.0, 10_000.0, 1.05, 0.0], [20.0, 250_000.0, 1.95, 3.0])
    else:
        x0s = [[lk, n0, 1.0] for lk in (-6.0, -3.0, 0.0, 3.0)]
        lo, hi = ([-20.0, 1.05, 0.0], [20.0, 1.95, 3.0])
    best, best_cost = None, np.inf
    for x0 in x0s:
        try:
            res = least_squares(resid, x0=x0, loss="soft_l1",
                                f_scale=huber_delta, bounds=(lo, hi),
                                max_nfev=800)
        except Exception:                       # noqa: BLE001
            continue
        if res.cost < best_cost:
            best, best_cost = res, res.cost
    if best is None:
        raise RuntimeError("the kinetic fit did not converge from any start")
    log_k, Ea, n, a = unpack(best.x)
    p = SulfurKernelParams(float(np.exp(log_k)), float(Ea), float(n), float(a),
                           T_ref_K, rho)
    pred = predict_sulfur(S_in[m], T[m], pH2[m], lh[m], p)
    p.fit_report = {
        "n_rows": int(m.sum()), "T_ref_C": T_ref, "cost": float(best_cost),
        "MAE_train": float(np.nanmean(np.abs(pred - y[m]))),
        "loss": "soft_l1 on log-sulfur (robust to off-spec lab results)",
        "status": "[DOP] lumped k_ref absorbs W_cat/V_L and the absolute LHSV; "
                  "only relative feed rate is identifiable from this package.",
        "Ea_source": ("[LIT] fixed at the literature value"
                      if Ea_fixed is not None else "[DOP] fitted"),
        "Ea_J_mol": float(Ea), "n": float(n), "a": float(a),
    }
    return p


def dS_dT(S_in_mgkg, T_C, pH2_MPa, lhsv_rel, p: SulfurKernelParams,
          h: float = 0.1) -> np.ndarray:
    """d S_out / d T_WABT  [mg/kg per degC] -- must be <= 0 (eq 13-15)."""
    return (predict_sulfur(S_in_mgkg, np.asarray(T_C, float) + h, pH2_MPa,
                           lhsv_rel, p)
            - predict_sulfur(S_in_mgkg, np.asarray(T_C, float) - h, pH2_MPa,
                             lhsv_rel, p)) / (2 * h)


def dS_dSin(S_in_mgkg, T_C, pH2_MPa, lhsv_rel, p: SulfurKernelParams,
            h: float = 1.0) -> np.ndarray:
    """d S_out / d S_in -- the feed-sulfur sensitivity used in eq 64."""
    S = np.asarray(S_in_mgkg, float)
    return (predict_sulfur(S + h, T_C, pH2_MPa, lhsv_rel, p)
            - predict_sulfur(S - h, T_C, pH2_MPa, lhsv_rel, p)) / (2 * h)


def dS_dlhsv(S_in_mgkg, T_C, pH2_MPa, lhsv_rel, p: SulfurKernelParams,
             h: float = 0.01) -> np.ndarray:
    """d S_out / d LHSV_rel -- the feed-rate sensitivity."""
    lh = np.asarray(lhsv_rel, float)
    return (predict_sulfur(S_in_mgkg, T_C, pH2_MPa, lh + h, p)
            - predict_sulfur(S_in_mgkg, T_C, pH2_MPa, lh - h, p)) / (2 * h)
