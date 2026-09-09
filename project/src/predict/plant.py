"""The assembled quality predictor -- model_spec.md §26, eq (55) + §24 eq (51).

    y_hat(t) = g_phi(u_tilde, d)      physical kernel (eq 16 via sulfur_kernel)
             + r_psi(u_tilde, d)      sign-constrained robust residual
             + eta(t)                 offset anchored on the laboratory result

The offset is what turns a modest steady-state model into a usable one-step
predictor: between lab results it is held and its variance grows (eq 52), and
every accepted LIMS value re-anchors it (eq 51).  Running the predictor over a
time range therefore requires replaying the sources in order, which is what
:func:`replay` does -- strictly causally, using at time t only what was
available at t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.estimation.fusion import BiasFilter, SourceRank, SourceReading, fuse
from src.kernels import sulfur_kernel as SK


#: Logical names of the kernel inputs, in the order sulfur_kernel expects them.
KERNEL_INPUT_ORDER = ("S_in", "T_WABT", "pH2", "LHSV")


@dataclass
class PlantPredictor:
    """eq 55 for the product sulfur.

    The kernel reads the **shaped** inputs ``kin:*`` of eq 32-33 (§26), not the
    instantaneous measurements: a set-point change reaches the product two to
    three hours later on this unit, and pretending otherwise would make every
    recommendation claim an immediate effect it cannot have.
    """
    kernel: SK.SulfurKernelParams
    residual_names: list[str]
    residual_theta: np.ndarray
    residual_intercept: float
    channel_keys: list[str]

    # ------------------------------------------------------------------
    @staticmethod
    def _kin(src, name: str):
        """Shaped kernel input, falling back to the raw column if absent."""
        from src.ident.channels import KERNEL_INPUT_DRIVERS, KIN
        col = f"{KIN}{name}"
        if col in src:
            return src[col]
        raw = next((c for c, (n, _) in KERNEL_INPUT_DRIVERS.items()
                    if n == name), None)
        if raw is not None and raw in src:
            return src[raw]
        raise KeyError(f"neither {col} nor a raw source for {name} is present")

    def kernel_value(self, row: pd.Series | dict,
                     utilde_row: pd.Series | dict | None = None) -> float:
        src = utilde_row if utilde_row is not None else row
        try:
            args = [float(self._kin(src, n)) for n in KERNEL_INPUT_ORDER]
        except KeyError:
            args = [float(self._kin(row, n)) for n in KERNEL_INPUT_ORDER]
        return float(SK.predict_sulfur(*args, self.kernel))

    def kernel_series(self, frame: pd.DataFrame,
                      utilde: pd.DataFrame | None = None) -> np.ndarray:
        src = utilde if utilde is not None else frame
        try:
            args = [self._kin(src, n).to_numpy(float)
                    for n in KERNEL_INPUT_ORDER]
        except KeyError:
            args = [self._kin(frame, n).to_numpy(float)
                    for n in KERNEL_INPUT_ORDER]
        return SK.predict_sulfur(*args, self.kernel)

    def residual(self, X) -> np.ndarray:
        """r_psi on a raw feature matrix ordered like ``residual_names``."""
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != len(self.residual_names):
            raise ValueError(
                f"expected {len(self.residual_names)} features, got {X.shape[1]}")
        return X @ np.asarray(self.residual_theta, float) + self.residual_intercept

    def residual_series(self, utilde: pd.DataFrame) -> np.ndarray:
        return self.residual(utilde[self.residual_names].to_numpy(float))

    def model_series(self, frame: pd.DataFrame, utilde: pd.DataFrame
                     ) -> np.ndarray:
        return (self.kernel_series(frame, utilde)
                + self.residual_series(utilde))

    # ------------------------------------------------------------------
    def sensitivity(self, row: pd.Series, key: str, delta: float,
                    utilde_row: pd.Series | None = None) -> float:
        """d y_hat / d u_key at the operating point, kernel + residual.

        The kernel carries the kinetic effect of temperature, pressure and feed
        rate; the residual adds its (sign-constrained) correction.  Computed by
        a central difference so a non-linear kernel is handled exactly.
        """
        base_res = 0.0
        name = f"{key}~lag"
        if name in self.residual_names:
            base_res = float(self.residual_theta[self.residual_names.index(name)])
        k = _kernel_sensitivity(self, row, key, delta, utilde_row)
        return k + base_res


def _kernel_sensitivity(pred: PlantPredictor, row, key: str, delta: float,
                        utilde_row=None) -> float:
    """Kernel sensitivity by perturbing the shaped input the key drives.

    This is the *steady-state* gain of a sustained move: the shaping of eq 33
    has unit static gain, so a held change of ``key`` eventually moves the
    kernel input one-for-one.  The transient is reported separately by
    ``horizon_response``.
    """
    try:
        S_in, T, p, lh = [
            float(pred._kin(utilde_row if utilde_row is not None else row, n))
            for n in KERNEL_INPUT_ORDER]
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not np.isfinite([S_in, T, p, lh]).all():
        return 0.0
    if key == "HDS:T5":
        # T_WABT = (1/3) T6 + (2/3) T5  ->  dT_WABT/dT5 = 2/3
        return float(SK.dS_dT(S_in, T, p, lh, pred.kernel)) * (2.0 / 3.0)
    if key == "HDS:P3":
        # p_H2 proxy = 0.8 * P3   (eq 17, chi_H2 = 0.8 [DOP])
        h = 0.01
        return float((SK.predict_sulfur(S_in, T, p + 0.8 * h, lh, pred.kernel)
                      - SK.predict_sulfur(S_in, T, p - 0.8 * h, lh, pred.kernel))
                     / (2 * h))
    if key == "HDS:F26":
        # LHSV_rel = F26 / median(F26)
        scale = lh / max(float(row["HDS:F26"]), 1e-9)
        return float(SK.dS_dlhsv(S_in, T, p, lh, pred.kernel)) * scale
    return 0.0


def horizon_response(pred: PlantPredictor, row, utilde_row, key: str,
                     du: float, dyn: dict, Ts_s: float, H: int) -> dict:
    """Sulfur trajectory over the horizon for a *held* change of one channel.

    A recommendation that quotes only the steady-state effect is misleading on
    a unit whose quality responds two to three hours after the move.  This
    propagates ``du`` through the identified FOPTD of the driving channel
    (eq 33) and evaluates the kernel at every step, so the report can state
    both the eventual effect and when it starts to appear.
    """
    from src.ident.channels import KERNEL_INPUT_DRIVERS
    from src.features.lag import first_order_filter, pure_delay

    target = next((n for c, (n, drv) in KERNEL_INPUT_DRIVERS.items()
                   if drv == key), None)
    if target is None:
        return {"available": False}
    d = dyn.get(key)
    if d is None:
        return {"available": False}
    try:
        base = {n: float(pred._kin(utilde_row, n)) for n in KERNEL_INPUT_ORDER}
    except (KeyError, TypeError, ValueError):
        return {"available": False}
    if not np.isfinite(list(base.values())).all():
        return {"available": False}

    # scale du in the channel's own units to the kernel input it drives
    scale = 1.0
    if key == "HDS:T5":
        scale = 2.0 / 3.0                      # dT_WABT / dT5
    elif key == "HDS:P3":
        scale = 0.8                            # p_H2 = chi_H2 * P3
    elif key == "HDS:F26":
        f26 = float(row.get("HDS:F26", np.nan))
        scale = (base["LHSV"] / f26) if np.isfinite(f26) and f26 > 0 else 0.0

    step = np.full(H, du * scale)
    shaped = first_order_filter(
        pure_delay(step, int(d.delay_samples), fill=0.0),
        d.tau_samples * Ts_s, Ts_s)
    y = np.empty(H)
    for k in range(H):
        args = dict(base)
        args[target] = base[target] + float(shaped[k])
        y[k] = float(SK.predict_sulfur(*[args[n] for n in KERNEL_INPUT_ORDER],
                                       pred.kernel))
    y0 = float(SK.predict_sulfur(*[base[n] for n in KERNEL_INPUT_ORDER],
                                 pred.kernel))
    delta = y - y0
    total = float(delta[-1])
    reached = np.nan
    if abs(total) > 1e-9:
        idx = np.where(np.abs(delta) >= 0.632 * abs(total))[0]
        if idx.size:
            reached = float((idx[0] + 1) * Ts_s / 3600.0)
    return {"available": True, "channel": key, "kernel_input": target,
            "delay_h": d.delay_h, "tau_h": d.tau_h,
            "y0_mgkg": y0, "trajectory_mgkg": y.tolist(),
            "delta_at_horizon_mgkg": total,
            "horizon_h": H * Ts_s / 3600.0,
            "time_to_63pct_h": reached,
            "note": "effect of holding the move; the quality response is "
                    "delayed by the identified transport time (eq 32-33)"}


@dataclass
class ReplayResult:
    frame: pd.DataFrame
    n_anchors: int
    n_rejected: int
    log: list = field(default_factory=list)


def replay(pred: PlantPredictor, frame: pd.DataFrame, utilde: pd.DataFrame,
           lab_col: str = "LIMS:sulfur", pak_col: str = "PAK:sulfur",
           q_bias_per_h: float = 0.02, r_lims: float = 0.25,
           r_pak: float = 4.0, tau_lims_max_h: float = 72.0,
           tau_pak_max_h: float = 2.0, gate_p: float = 0.99,
           sigma_gate: float = 6.0, use_pak: bool = True) -> ReplayResult:
    """Causal replay of eq 51-52 over the whole frame.

    At every row the model value is computed; at every row where a *new*
    laboratory (or ПАК) result becomes available the offset is updated.  The
    prediction stored on a lab row is the one made **before** that result was
    used, so it is a genuine out-of-sample one-step forecast.
    """
    model = pred.model_series(frame, utilde)
    t_h = ((frame["date"] - frame["date"].iloc[0]).dt.total_seconds()
           .to_numpy() / 3600.0)
    lab = frame[lab_col].to_numpy(float)
    lab_new = frame[f"{lab_col}__new"].to_numpy(bool)
    lab_age = frame[f"{lab_col}__age_h"].to_numpy(float)
    has_pak = use_pak and pak_col in frame.columns
    pak = frame[pak_col].to_numpy(float) if has_pak else None
    pak_new = frame[f"{pak_col}__new"].to_numpy(bool) if has_pak else None
    pak_stale = (frame[f"{pak_col}__stale"].to_numpy(bool)
                 if has_pak and f"{pak_col}__stale" in frame.columns else None)

    n = len(frame)
    eta_out = np.zeros(n)
    P_out = np.zeros(n)
    src_out = np.array([""] * n, dtype=object)
    n_anchor = n_reject = 0
    log: list[dict] = []

    # Only the rows where a source delivers a value need the scalar recursion;
    # between them eta is held and P grows linearly (eq 52), which is filled in
    # a vectorised slice.
    new_any = lab_new & np.isfinite(lab)
    if has_pak:
        new_any = new_any | (pak_new & np.isfinite(pak))
    events = np.where(new_any)[0]

    eta, P, t_last, seg = 0.0, max(r_lims * 4.0, 1.0), t_h[0], 0
    for k in events:
        # hold eta and grow P over the segment ending at this event (eq 52)
        eta_out[seg:k + 1] = eta
        P_out[seg:k + 1] = P + q_bias_per_h * np.clip(t_h[seg:k + 1] - t_last,
                                                      0.0, None)
        P_pre = float(P_out[k])
        seg = k + 1
        t_last = t_h[k]
        P = P_pre
        m = model[k]

        # source hierarchy: a fresh laboratory result always wins (§24)
        if lab_new[k] and np.isfinite(lab[k]) and lab_age[k] <= tau_lims_max_h:
            value, R, name = float(lab[k]), r_lims, "LIMS"
        elif (has_pak and pak_new[k] and np.isfinite(pak[k])
              and not (pak_stale is not None and pak_stale[k])):
            value, R, name = float(pak[k]), r_pak, "PAK"
        else:
            continue
        if not np.isfinite(m):
            continue

        innov = value - (m + eta)
        S = P_pre + R
        if S <= 0 or innov * innov / S > sigma_gate * sigma_gate:   # eq 49 gate
            n_reject += 1
            if len(log) < 5000:
                log.append({"k": int(k), "t": str(frame["date"].iloc[k]),
                            "source": name, "accepted": False,
                            "reason": f"innovation gate: |r|={abs(innov):.2f} "
                                      f"vs sqrt(S)={np.sqrt(max(S, 0.0)):.2f}"})
            continue

        K = P_pre / (P_pre + R)               # eq 51
        eta = eta + K * innov
        P = (1.0 - K) * P_pre
        n_anchor += 1
        src_out[k] = name
        if len(log) < 5000:
            log.append({"k": int(k), "t": str(frame["date"].iloc[k]),
                        "source": name, "accepted": True,
                        "innovation": float(innov), "gain": float(K),
                        "eta": float(eta), "P_eta": float(P)})

    eta_out[seg:] = eta
    P_out[seg:] = P + q_bias_per_h * np.clip(t_h[seg:] - t_last, 0.0, None)
    # the value stored on an event row is the one available *before* that
    # measurement was used, so a lab row carries a genuine out-of-sample forecast
    pred_out = np.where(np.isfinite(model), model + eta_out, np.nan)

    res = frame[["date"]].copy()
    res["model"] = model
    res["eta"] = eta_out
    res["P_eta"] = P_out
    res["y_hat"] = pred_out
    res["anchor_source"] = src_out
    res[lab_col] = lab
    res[f"{lab_col}__new"] = lab_new
    res[f"{lab_col}__age_h"] = lab_age
    return ReplayResult(res, n_anchor, n_reject, log)
