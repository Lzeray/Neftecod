"""Catalyst deactivation proxy -- model_spec.md §30, eq (62).

    (62)  dT_norm(t) = T_WABT(t) - T_WABT_hat(S, LHSV, p_H2)
          trend: d/dt dT_norm

§30 is explicit: this is a *hypothesis* about deactivation/offset (eq 47), not
a measured catalyst activity.  Everything this module returns is labelled as
such and enters only as (a) a term of the severity index and (b) a tightening
of the upper bound on the T_WABT set point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NormalisationModel:
    """T_WABT_hat(S, LHSV, p_H2) -- the regression subtracted in eq 62."""
    coef: np.ndarray
    names: list[str]
    intercept: float
    r2: float

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != len(self.names):
            raise ValueError("feature count mismatch")
        return X @ self.coef + self.intercept


def fit_normalisation(X, T_wabt, names: list[str]) -> NormalisationModel:
    """Least-squares T_WABT_hat on (S, LHSV, p_H2) -- the eq 62 reference."""
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(T_wabt, float).ravel()
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if m.sum() < X.shape[1] + 2:
        raise ValueError("not enough complete rows to fit the normalisation")
    A = np.column_stack([X[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    pred = A @ coef
    ss_res = float(((y[m] - pred) ** 2).sum())
    ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return NormalisationModel(coef[:-1], names, float(coef[-1]), r2)


def delta_T_norm(T_wabt, X, model: NormalisationModel) -> np.ndarray:
    """dT_norm of eq (62)."""
    return np.asarray(T_wabt, float).ravel() - model.predict(X)


@dataclass
class DeactivationReport:
    slope_c_per_1000h: float
    p_value: float
    n: float
    span_h: float
    significant: bool
    status: str = "[DOP, hypothesis]"
    note: str = ("Normalised-temperature drift (eq 62). This is an offset "
                 "hypothesis (eq 47), NOT a measured catalyst activity.")

    def as_dict(self) -> dict:
        return {"slope_c_per_1000h": self.slope_c_per_1000h,
                "p_value": self.p_value, "n": self.n, "span_h": self.span_h,
                "significant": self.significant, "status": self.status,
                "note": self.note}


def drift_trend(t_h, dT_norm, alpha: float = 0.05) -> DeactivationReport:
    """Linear trend of dT_norm, reported per 1000 h (eq 62 second half)."""
    t = np.asarray(t_h, float).ravel()
    y = np.asarray(dT_norm, float).ravel()
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 10:
        raise ValueError("need at least 10 points to estimate a trend")
    res = stats.linregress(t[m], y[m])
    span = float(t[m].max() - t[m].min())
    return DeactivationReport(
        slope_c_per_1000h=float(res.slope * 1000.0),
        p_value=float(res.pvalue), n=float(m.sum()), span_h=span,
        significant=bool(res.pvalue < alpha and res.slope > 0),
    )


def tighten_temperature_bound(u_hi: float, report: DeactivationReport,
                              margin_per_c: float = 1.0,
                              max_tighten_c: float = 5.0) -> tuple[float, str]:
    """Lower the T_WABT upper bound when a deactivation drift is detected.

    The tightening is proportional to the observed drift and capped, and the
    reason is returned so it appears verbatim in the operator report.
    """
    if not report.significant:
        return float(u_hi), "no significant normalised-temperature drift"
    tighten = min(max(report.slope_c_per_1000h, 0.0) * margin_per_c,
                  max_tighten_c)
    return (float(u_hi - tighten),
            f"[DOP] T_WABT upper bound tightened by {tighten:.2f} degC: "
            f"normalised temperature drifts +{report.slope_c_per_1000h:.2f} "
            f"degC/1000 h (p={report.p_value:.3g}) -- deactivation hypothesis "
            f"(eq 62), not a measured activity")
