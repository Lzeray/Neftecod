"""Quality agent -- ТЗ §3; model_spec.md §26-28, blocks C/D/E/F.

Returns the predicted quality, its uncertainty (eq 58 *and* the set-based
eq 60), the probability of violating each specification, the sensitivity of the
product sulfur to every manipulated channel, and an explicit confidence label
derived from data freshness and the distance from the identified domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import domain
from src.agents.contracts import PlantState, QualityAssessment
from src.data import staleness
from src.optimize.chance import check_max
from src.predict.plant import PlantPredictor
from src.predict.uncertainty import UncertaintyModel, set_error_bound, z_quantile


class QualityAgent:
    name = "quality"

    def __init__(self, predictor: PlantPredictor, unc: UncertaintyModel,
                 cfg, state_space=None, w_half=None, v_half=None,
                 channel_dynamics: dict | None = None):
        self.pred = predictor
        self.unc = unc
        self.cfg = cfg
        self.ss = state_space
        self.w_half = w_half
        self.v_half = v_half
        self.dyn = channel_dynamics or {}

    # ------------------------------------------------------------------
    def assess(self, state: PlantState, row: pd.Series, utilde_row: pd.Series,
               eta: float, P_eta: float) -> QualityAssessment:
        alpha = float(self.cfg["spec"]["alpha"])
        S_max = float(self.cfg["spec"]["S_max_mgkg"])

        kernel = self.pred.kernel_value(row, utilde_row)
        X = np.array([[float(utilde_row.get(n, np.nan))
                       for n in self.pred.residual_names]])
        resid = float(self.pred.residual(X)[0]) if np.isfinite(X).all() else 0.0
        y_s = kernel + resid + float(eta)

        tau = float(state.freshness.get("LIMS:sulfur", np.inf))
        tau_eff = tau if np.isfinite(tau) else float(
            self.cfg["spec"]["tau_lims_max_h"]) * 2.0
        # leverage is measured in the kernel's (shaped) input space
        from src.predict.plant import KERNEL_INPUT_ORDER
        try:
            x_lev = np.array([float(self.pred._kin(utilde_row, n))
                              for n in KERNEL_INPUT_ORDER])
        except (KeyError, TypeError, ValueError):
            x_lev = None
        if x_lev is not None and not np.isfinite(x_lev).all():
            x_lev = None
        sigma = self.unc.sigma_pred(tau_h=min(tau_eff, 1e4), P_eta=P_eta,
                                    x=x_lev)

        # set-based bound (eq 59-60) -- the conservative companion of eq 58
        eps = float("nan")
        if self.ss is not None and self.w_half is not None:
            e = set_error_bound(self.ss.A, self.ss.C, self.w_half,
                                self.v_half, int(self.cfg["horizon"]["H"]))
            eps = float(e[-1, 0])

        chk = check_max(y_s, sigma, S_max, alpha)
        sens = {k: float(self.pred.sensitivity(row, k, 1.0, utilde_row))
                for k in domain.MANIPULATED_KEYS}
        grad_d = self._grad_d(row, utilde_row)

        extrap = self.unc.extrapolation(
            x_lev,
            threshold=(float(self.cfg["uncertainty"].get(
                "extrapolation_threshold_per_dim", 4.0))
                * (len(x_lev) if x_lev is not None else 1))
        ) if x_lev is not None else {"available": False}
        conf, notes = self._confidence(tau, sigma, state, extrap)
        horizon = self._horizon(row, utilde_row)
        return QualityAssessment(
            timestamp=state.timestamp,
            y_hat={"sulfur": y_s, "sulfur_kernel": kernel,
                   "sulfur_residual": resid, "sulfur_offset": float(eta)},
            sigma_pred={"sulfur": sigma},
            eps_plus={"sulfur": eps},
            p_violation={"sulfur": chk.p_violation},
            backoff={"sulfur": chk.backoff},
            sensitivities=sens, grad_d=grad_d, confidence=conf,
            data_quality={
                "lims_age_h": tau, "P_eta": P_eta,
                "leverage": (self.unc.leverage(x_lev)
                             if x_lev is not None else float("nan")),
                "extrapolation": extrap,
                "z": z_quantile(alpha),
                "pak_stale": bool(state.flags.get("PAK:sulfur__stale", False)),
                "pak_outlier": bool(state.flags.get("PAK:sulfur__outlier", False)),
            },
            sources=state.flags.get("_sources", []) if isinstance(
                state.flags.get("_sources"), list) else [],
            horizon=horizon, notes=notes)

    # ------------------------------------------------------------------
    def _horizon(self, row, utilde_row) -> dict:
        """When a held move actually reaches the product (eq 32-33).

        The steady-state sensitivity alone would let a report claim an
        immediate effect; on this unit the identified transport delay is 2-6 h,
        which is longer than the decision horizon for some channels.
        """
        from src.predict.plant import horizon_response
        if not self.dyn or utilde_row is None:
            return {}
        H = int(self.cfg["horizon"]["H"])
        out = {}
        for key in domain.MANIPULATED_KEYS:
            if key not in self.dyn:
                continue
            hr = horizon_response(self.pred, row, utilde_row, key, 1.0,
                                  self.dyn, float(self.cfg.Ts), H)
            if hr.get("available"):
                out[key] = {
                    "delay_h": hr["delay_h"], "tau_h": hr["tau_h"],
                    "time_to_63pct_h": hr["time_to_63pct_h"],
                    "dS_per_unit_at_horizon": hr["delta_at_horizon_mgkg"],
                    "horizon_h": hr["horizon_h"],
                }
        return out

    def _grad_d(self, row, utilde_row=None) -> dict[str, float]:
        """d S / d d for the robust counterpart (eq 64)."""
        from src.kernels import sulfur_kernel as SK
        from src.predict.plant import KERNEL_INPUT_ORDER
        try:
            args = tuple(float(self.pred._kin(utilde_row, n))
                         for n in KERNEL_INPUT_ORDER)
        except (KeyError, TypeError, ValueError):
            return {}
        if not np.isfinite(args).all():
            return {}
        return {"feed_sulfur_mgkg": float(SK.dS_dSin(*args, self.pred.kernel))}

    def _confidence(self, tau_h: float, sigma: float, state: PlantState,
                    extrap: dict | None = None):
        notes = []
        extrap = extrap or {}
        if extrap.get("outside"):
            z = extrap.get("per_dimension_z", [])
            worst = (int(np.argmax(np.abs(z))) if len(z) else None)
            names = ["feed sulfur", "T_WABT", "p_H2 proxy", "relative LHSV"]
            which = (names[worst] if worst is not None and worst < len(names)
                     else "an input")
            notes.append(
                f"the operating point is outside the domain the kernel was "
                f"identified on (Mahalanobis d^2 = {extrap['mahalanobis2']:.1f} "
                f"vs threshold {extrap['threshold']:.0f}); the largest "
                f"deviation is {which} at {z[worst]:+.1f} sigma")
        tau_max = float(self.cfg["spec"]["tau_lims_max_h"])
        s_max = float(self.cfg["spec"]["sigma_max"])
        score = 1 if extrap.get("outside") else 0
        if not np.isfinite(tau_h) or tau_h > tau_max:
            notes.append(f"the last laboratory sulfur result is "
                         f"{tau_h:.1f} h old (limit {tau_max:.0f} h)")
            score += 2
        elif tau_h > 0.5 * tau_max:
            notes.append(f"the laboratory result is ageing ({tau_h:.1f} h)")
            score += 1
        if sigma > s_max:
            notes.append(f"sigma_pred={sigma:.2f} mg/kg exceeds the confidence "
                         f"threshold {s_max:.2f}")
            score += 2
        elif sigma > 0.7 * s_max:
            score += 1
        if state.flags.get("PAK:sulfur__stale"):
            notes.append("the stream sulfur analyser is frozen (zero variance "
                         "over the staleness window)")
            score += 1
        return ("high" if score == 0 else "medium" if score <= 2 else "low",
                notes)
