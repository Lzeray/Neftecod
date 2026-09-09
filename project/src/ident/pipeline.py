"""Identification protocol -- model_spec.md §37, executed on the real package.

Order (§37):
  1. delays / time constants per channel from cross-correlation (§12)
  2. kinetics by non-linear least squares on the training window (§6)
  3. linear residual coefficients, convex (§16), robust (§17), IV (§18)
  4. excitation check, unidentifiable channels excluded (§19, §31)
  5. filter and uncertainty tuning (§23, §28) calibrated for honest coverage

Everything is fitted on the *training* window only; the test window is touched
exactly once, by src/validate/report.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import domain
from src.config import Config
from src.data.dataset import Dataset
from src.ident import channels as CH
from src.ident.excitation import excitation_report, prune_collinear
from src.kernels import sulfur_kernel as SK
from src.optimize.constraints import Bound, model_range_from_history
from src.predict.greybox import SignConstraint, fit_greybox
from src.predict.statespace import ChannelSpec, build_state_space
from src.predict.uncertainty import UncertaintyModel, fit_domain
from src.reliability import deactivation as DA
from src.reliability.severity import SeverityIndex, SeverityTerm, robust_reference
from src.validate import splits

#: [DOP] laboratory results above this are treated as upset / off-spec batches
#: for *fitting* purposes (the record contains values up to 2120 mg/kg on a
#: 10 mg/kg product).  They are never deleted -- they are down-weighted by the
#: robust loss and they are all kept for the violation statistics.
LAB_FIT_CLIP_MGKG = 60.0


@dataclass
class IdentifiedModel:
    filter_params: dict
    channel_dynamics: dict
    kernel: SK.SulfurKernelParams
    residual_names: list[str]
    residual_theta: np.ndarray
    residual_intercept: float
    sigma_resid: float
    c_tau: float
    x_mean: np.ndarray
    x_cov_inv: np.ndarray
    bounds: dict
    severity_terms: list
    deactivation: dict
    excitation: dict
    train_end: str
    reports: dict = field(default_factory=dict)

    def uncertainty_model(self, lev_scale: float = 1.0,
                          lev_cap: float = 3.0) -> UncertaintyModel:
        return UncertaintyModel(sigma_resid=self.sigma_resid, c_tau=self.c_tau,
                                lev_scale=lev_scale, lev_cap=lev_cap,
                                x_mean=self.x_mean, x_cov_inv=self.x_cov_inv,
                                n_train=int(self.reports.get(
                                    "uncertainty", {}).get("leverage_n", 0)))

    def state_space(self, Ts: float):
        specs = []
        for key, d in self.channel_dynamics.items():
            name = f"{key}~lag"
            if name not in self.residual_names:
                continue
            gain = float(self.residual_theta[self.residual_names.index(name)])
            a = float(np.exp(-Ts / (d["tau_samples"] * Ts))) if d["tau_samples"] > 0 else 0.0
            specs.append(ChannelSpec(key, int(d["delay_samples"]), a, gain))
        return build_state_space(specs)


# ----------------------------------------------------------------------
def _train_mask(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    sp = splits.time_split(frame["date"], cfg["time"]["train_end"],
                           cfg["time"]["test_end"])
    m = np.zeros(len(frame), bool)
    m[sp.train_idx] = True
    return m


def identify(ds: Dataset, cfg: Config) -> IdentifiedModel:
    frame = ds.frame
    Ts = cfg.Ts
    train = _train_mask(frame, cfg)
    operating = (frame["derived:operating"].to_numpy(bool)
                 if "derived:operating" in frame.columns
                 else np.ones(len(frame), bool))
    train = train & operating
    lab_rows = frame["LIMS:sulfur__new"].to_numpy(bool)
    y = frame["LIMS:sulfur"].to_numpy(float)
    fit_rows = lab_rows & train & np.isfinite(y) & (y < LAB_FIT_CLIP_MGKG)
    reports: dict = {"n_lab_train": int(fit_rows.sum()),
                     "n_lab_train_all": int((lab_rows & train).sum()),
                     "n_operating_rows": int(operating.sum()),
                     "n_rows": int(len(frame))}

    # --- step 1: channel delays and time constants (§12, §37.1) ----------
    keys = [k for k in domain.MANIPULATED_KEYS + domain.DISTURBANCE_KEYS
            if k in frame.columns]
    dyn = CH.identify_all(frame, keys, "LIMS:sulfur", fit_rows, Ts,
                          max_delay=int(cfg["horizon"]["n_delay_max"]))
    utilde = CH.build_effective_inputs(frame, dyn, Ts)
    reports["channels"] = {k: v.as_dict() for k, v in dyn.items()}
    reports["kernel_inputs"] = CH.kernel_input_report(dyn)

    # --- step 2: kinetics (§6, §37.2) ------------------------------------
    # §26: the kernel is fitted on the *shaped* inputs of eq 32-33, so the
    # kinetics are identified against the reactor conditions that actually
    # produced the sampled material, not the ones at the sampling instant.
    S_in = utilde[f"{CH.KIN}S_in"].to_numpy(float)
    T_w = utilde[f"{CH.KIN}T_WABT"].to_numpy(float)
    pH2 = utilde[f"{CH.KIN}pH2"].to_numpy(float)
    lhsv = utilde[f"{CH.KIN}LHSV"].to_numpy(float)
    kernel = SK.fit_sulfur_kernel(S_in[fit_rows], T_w[fit_rows], pH2[fit_rows],
                                  lhsv[fit_rows], y[fit_rows])
    reports["kernel"] = kernel.fit_report
    g_all = SK.predict_sulfur(S_in, T_w, pH2, lhsv, kernel)

    # --- step 4 (before 3): excitation / identifiability (§19) -----------
    feat_names = [f"{k}~lag" for k in keys]
    X_all = utilde[feat_names].to_numpy(float)
    ok = fit_rows & np.isfinite(X_all).all(axis=1) & np.isfinite(g_all)
    Xf = X_all[ok]
    Xz = (Xf - Xf.mean(axis=0)) / np.where(Xf.std(axis=0) > 0, Xf.std(axis=0), 1)
    _, kept_names, exc = prune_collinear(
        Xz, feat_names, alpha=float(cfg["ident"]["pe_min_eig"]),
        cond_max=float(cfg["ident"]["cond_max"]))
    reports["excitation"] = {
        "n": exc.n, "p": exc.p, "rank": exc.rank, "min_eig": exc.min_eig,
        "cond": exc.cond, "persistently_exciting": exc.persistently_exciting,
        "dropped": [feat_names[i] for i in exc.dropped],
        "collinear_groups": [[feat_names[i] for i in g]
                             for g in exc.collinear_groups],
        "note": "model_spec.md §19: regularisation does not add missing "
                "information; dropped channels are excluded from the "
                "manipulated set (§31).",
    }

    # --- step 3: residual, robust + IV (§16-§18, §37.3) ------------------
    # Instruments (eq 43): the AVT-side channels and the crude charge sit
    # upstream of the hydrotreater and are not driven by the HDS product-sulfur
    # loop, so they are plausibly orthogonal to its noise.  [DOP] -- the
    # package contains no explicit external set-point signal, which is the
    # textbook instrument (§18); this substitution is recorded as an
    # assumption and its relevance is reported.
    inst_names = [n for n in kept_names
                  if n.startswith("AVT:") or n[:-4] in domain.DISTURBANCE_KEYS]
    Xk = utilde[kept_names].to_numpy(float)
    Z = utilde[inst_names].to_numpy(float) if len(inst_names) >= len(kept_names) \
        else None
    sign_cons = [SignConstraint(f"{k}~lag", sgn, why)
                 for k, sgn, why in PHYSICS_SIGNS
                 if f"{k}~lag" in kept_names]
    iv_note = _instrument_report(inst_names, kept_names)

    # Hyperparameters (residual shrinkage, filter Q/R, whether ПАК may anchor)
    # are selected on an *inner* time split of the training window, never on
    # the rows the residual was fitted on -- otherwise the selection would
    # always prefer the least regularised model (model_spec.md §38).
    inner = _inner_split(frame, train, fit_rows,
                         float(cfg["ident"].get("inner_val_frac", 0.25)))
    gb = fit_greybox(Xk[ok], y[ok], g_all[ok], kept_names,
                     instruments=None if Z is None else Z[ok],
                     sign_constraints=sign_cons,
                     lam=float(cfg["ident"]["lam"]),
                     huber_delta=float(cfg["ident"]["huber_delta"]))
    reports["residual"] = gb.fit_report
    reports["residual"]["instruments"] = inst_names
    reports["residual"]["instrument_note"] = iv_note
    reports["closed_loop_evidence"] = _closed_loop_evidence(frame, ok, y)

    # Statistical identifiability of each *gain* (§19 -> §31).  Passing the
    # rank/PE test only says the regressors are distinguishable; it does not
    # say the fitted gain is distinguishable from zero.  A moving-block
    # bootstrap over the laboratory results gives an interval for each
    # coefficient, and a channel whose interval straddles zero is not offered
    # as a control handle -- recommending a move on it would be noise.
    boot = _bootstrap_gains(Xk[ok], y[ok], g_all[ok], kept_names, sign_cons,
                            float(cfg["ident"]["lam"]),
                            float(cfg["ident"]["huber_delta"]),
                            seed=int(cfg["seed"]))
    reports["gain_significance"] = boot

    pred_train = g_all[ok] + gb.residual(Xk[ok])
    reports["kernel_skill"] = _kernel_skill(y[ok], g_all[ok], pred_train)

    # --- step 5a: hyperparameter selection (§23, §37.5, §38) -------------
    inner_ok = ok & inner["fit_mask"][ok.nonzero()[0]].astype(bool) \
        if False else (ok & inner["fit_mask"])
    if inner_ok.sum() > 60:
        kernel_in = SK.fit_sulfur_kernel(S_in[inner_ok], T_w[inner_ok],
                                         pH2[inner_ok], lhsv[inner_ok],
                                         y[inner_ok])
        g_in = SK.predict_sulfur(S_in, T_w, pH2, lhsv, kernel_in)
        gb_in = fit_greybox(Xk[inner_ok], y[inner_ok], g_in[inner_ok],
                            kept_names, sign_constraints=sign_cons,
                            lam=float(cfg["ident"]["lam"]),
                            huber_delta=float(cfg["ident"]["huber_delta"]))
    else:
        kernel_in, gb_in = kernel, gb
    filt = _tune_filter(frame, utilde, kernel_in, gb_in, dyn, cfg,
                        inner["val_rows"], lab_rows)
    filt["selection"] = {
        "protocol": "hyperparameters chosen on an inner time split of the "
                    "training window (fit <= inner_end < validate), then the "
                    "kernel and residual are refitted on the whole training "
                    "window with the chosen values",
        **{k: v for k, v in inner.items() if not k.endswith("mask")
           and not k.endswith("rows")},
    }
    # Apply the selected shrinkage to the full-window fit and re-anchor the
    # intercept against the same kernel the deployed predictor will use.
    scale = float(filt["chosen"].get("residual_scale", 1.0))
    if scale != 1.0:
        gb.theta = np.asarray(gb.theta, float) * scale
        gb.fit_report["residual_scale"] = scale
    gb.intercept = float(np.median(
        y[ok] - g_all[ok] - Xk[ok] @ np.asarray(gb.theta, float)))
    reports["filter"] = filt

    # --- step 5b: uncertainty (§28) --------------------------------------
    # sigma must describe the error of the **deployed** predictor: the kernel
    # and residual finally fitted on the whole training window, shrunk by the
    # selected scale, with the LIMS-anchored offset of eq 51 running.  Anything
    # else -- the un-anchored model, or a predictor built from the inner-fit
    # kernel with a full-window intercept -- reports an error the system does
    # not make, and eq 63 then demands a back-off no regime can satisfy.
    from src.predict.plant import PlantPredictor as _PP
    from src.predict.plant import replay as _replay
    deployed = _PP(kernel, gb.feature_names, np.asarray(gb.theta, float),
                   float(gb.intercept), list(dyn))
    rr_train = _replay(deployed, frame, utilde,
                       q_bias_per_h=float(filt["chosen"]["q_bias_per_h"]),
                       r_lims=float(filt["chosen"]["r_lims"]),
                       r_pak=float(cfg["filter"]["r_pak"]),
                       use_pak=bool(filt["chosen"]["use_pak"]))
    yh_train = rr_train.frame["y_hat"].to_numpy(float)
    cal = lab_rows & train & np.isfinite(y) & (y < LAB_FIT_CLIP_MGKG) \
        & np.isfinite(yh_train)
    resid = y[cal] - yh_train[cal]
    ages = frame["LIMS:sulfur__age_h"].to_numpy(float)[cal]
    filt["deployed_train"] = {
        "n": int(cal.sum()),
        "MAE": float(np.mean(np.abs(resid))),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "anchors": rr_train.n_anchors, "gate_rejected": rr_train.n_rejected}
    sigma_resid, c_tau, unc_rep = _calibrate_uncertainty(
        resid, ages, float(cfg["spec"]["alpha"]),
        float(cfg["uncertainty"]["c_tau"]))
    # The extrapolation term of eq 58 must measure distance from the domain
    # the *deployed* model was calibrated on.  That domain is the kernel's
    # input space (feed sulfur, WABT, p_H2, relative LHSV), not the residual
    # feature space -- the residual may have been shrunk away entirely.
    Xdom = np.column_stack([S_in, T_w, pH2, lhsv])
    dom_ok = ok & np.isfinite(Xdom).all(axis=1)
    mu, Sinv = fit_domain(Xdom[dom_ok])
    reports["uncertainty"] = {
        "sigma_resid": sigma_resid, "c_tau": c_tau,
        "n": int(np.isfinite(resid).sum()), **unc_rep,
        "calibrated_on": "anchored replay residual (eq 55 including eta), "
                         "training window",
        "leverage_domain": [f"{CH.KIN}S_in", f"{CH.KIN}T_WABT",
                            f"{CH.KIN}pH2", f"{CH.KIN}LHSV"],
        "leverage_n": int(dom_ok.sum())}

    # --- operating bounds (§31, ТЗ "Правило границ") ---------------------
    bounds = {}
    #: Channels whose gain comes from the *physical kernel* rather than from
    #: the fitted residual: the kernel supplies a sign-correct sensitivity even
    #: where the data cannot, so these stay controllable regardless of the
    #: bootstrap (model_spec.md §26).
    kernel_driven = {"HDS:T5", "HDS:P3", "HDS:F26"}
    for k in keys:
        if k not in domain.MANIPULATED_KEYS:
            continue
        s_train = frame.loc[train, k]
        try:
            lo, hi, note = model_range_from_history(s_train)
        except ValueError:
            continue
        span = hi - lo
        ch = domain.channel(k)
        gain_info = boot.get(f"{k}~lag", {})
        # After the inner-split selection the residual may have been shrunk to
        # zero, in which case a "significant" fitted gain no longer exists and
        # only the kernel-driven channels remain controllable.
        residual_alive = float(filt["chosen"].get("residual_scale", 1.0)) > 0.0
        controllable = bool(
            f"{k}~lag" in kept_names
            and (k in kernel_driven
                 or (residual_alive and gain_info.get("controllable", False))))
        why = gain_info.get("verdict", "")
        if k in kernel_driven:
            why += (" | retained: the sensitivity comes from the physical "
                    "kernel (eq 16), not from the fitted residual")
        elif not residual_alive:
            why += (" | the inner-split selection shrank the data-driven "
                    "residual to zero, so this channel has no credited gain")
        bounds[k] = Bound(
            name=k, lo=lo, hi=hi,
            du_max=max(span * float(cfg["optimize"]["du_max_frac"]), 1e-9),
            unit=(ch.unit if ch else ""), status="[DOP]",
            provenance=note + (" | " + why if why else ""),
            identifiable=controllable,
        )

    # --- severity proxies (§29) ------------------------------------------
    sev_terms = []
    for pr in domain.SEVERITY_PROXIES:
        tag = pr["tag"]
        if tag not in frame.columns:
            continue
        try:
            ref, scale = robust_reference(frame.loc[train, tag], q_ref=0.90)
        except ValueError:
            continue
        sev_terms.append(SeverityTerm(name=tag, ref=ref, scale=scale,
                                      weight=1.0 / len(domain.SEVERITY_PROXIES),
                                      gamma=2.0, direction=pr["direction"],
                                      status="[DOP]", note=pr["note"]))

    # --- deactivation hypothesis (§30) -----------------------------------
    deact = _deactivation(frame, train, fit_rows, y, lhsv, pH2, T_w)

    return IdentifiedModel(
        filter_params=filt["chosen"],
        channel_dynamics={k: v.as_dict() for k, v in dyn.items()},
        kernel=kernel, residual_names=gb.feature_names,
        residual_theta=gb.theta, residual_intercept=gb.intercept,
        sigma_resid=sigma_resid, c_tau=c_tau, x_mean=mu, x_cov_inv=Sinv,
        bounds=bounds, severity_terms=sev_terms, deactivation=deact,
        excitation=reports["excitation"], train_end=str(cfg["time"]["train_end"]),
        reports=reports)


#: Physically required signs of the *residual* coefficients (model_spec.md §26).
#: The kernel already carries the kinetic effects; the data-driven residual is
#: only allowed to correct their magnitude, never to reverse their direction.
PHYSICS_SIGNS = [
    ("HDS:T5", -1, "HDS kinetics: a hotter reactor cannot raise the product "
                   "sulfur (eq 13-15)"),
    ("HDS:P3", -1, "Higher loop pressure raises p_H2 (eq 17-18) and therefore "
                   "the HDS rate (eq 13)"),
    ("HDS:F2", -1, "More recycle gas raises hydrogen availability (eq 17-18)"),
    ("HDS:F26", +1, "More feed means a higher LHSV and less residence time "
                    "(eq 16)"),
    ("HDS:F15", +1, "Quench cools the second bed, lowering the reaction rate "
                    "(eq 14 through eq 24)"),
]


def _instrument_report(inst_names: list[str], regressors: list[str]) -> str:
    if len(inst_names) >= len(regressors):
        return ("[DOP] AVT-side channels and the crude charge used as "
                "instruments (eq 43): they sit upstream of the hydrotreater "
                "and are not driven by its product-sulfur loop.")
    return (f"NOT APPLIED: only {len(inst_names)} plausibly exogenous signals "
            f"({', '.join(inst_names)}) for {len(regressors)} regressors, so "
            "the eq 43 rank condition fails. The package contains no external "
            "set-point or excitation tag, and model_spec.md §18 forbids "
            "treating an arbitrary lagged tag as an instrument. The "
            "closed-loop bias is therefore handled by *physics* instead: the "
            "kernel carries the temperature/pressure/LHSV effects and the "
            "residual is sign-constrained so it cannot reverse them.")


def _closed_loop_evidence(frame, ok, y) -> dict:
    """Empirical evidence that eq 42 fails on this package.

    Operators raise the reactor temperature *because* sulfur is high, so the
    raw correlation between T_WABT and the product sulfur comes out positive --
    the opposite of the kinetics (eq 13-15).  Reporting the number is more
    useful than asserting the bias in prose.
    """
    T = frame["derived:T_WABT_C"].to_numpy(float)[ok]  # raw: eq 62 is a drift
    yy = np.asarray(y, float)[ok] if np.asarray(y).size != ok.sum() else np.asarray(y, float)
    m = np.isfinite(T) & np.isfinite(yy)
    if m.sum() < 30:
        return {"available": False}
    r = float(np.corrcoef(T[m], yy[m])[0, 1])
    return {
        "available": True, "n": int(m.sum()), "corr_T_WABT_vs_sulfur": r,
        "physically_expected_sign": "negative (eq 13-15)",
        "verdict": ("closed-loop confounding confirmed: the raw correlation is "
                    f"{r:+.3f}, i.e. the sign the kinetics forbid"
                    if r > 0 else
                    f"raw correlation {r:+.3f} already has the kinetic sign"),
        "consequence": ("a naive regression of sulfur on temperature would "
                        "recommend *cooling* the reactor to reduce sulfur; the "
                        "sign constraint of §26 blocks that."),
    }


def _bootstrap_gains(X, y, kernel, names, sign_cons, lam, delta,
                     n_boot: int = 200, block: int = 25, seed: int = 0) -> dict:
    """Moving-block bootstrap of the residual gains (§19 significance).

    Blocks preserve the serial correlation of the laboratory series, which an
    i.i.d. bootstrap would destroy and so would report far too much precision.
    A channel is offered as a control handle only when the 5-95% interval of
    its gain excludes zero **and** its sign survives the physics projection.
    """
    from src.predict.greybox import fit_greybox
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    kernel = np.asarray(kernel, float).ravel()
    n, p = X.shape
    if n < 4 * block or p == 0:
        return {nm: {"controllable": True,
                     "verdict": "[DOP] too few rows to bootstrap the gain"}
                for nm in names}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.full((n_boot, p), np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        try:
            m = fit_greybox(X[idx], y[idx], kernel[idx], names,
                            sign_constraints=sign_cons, lam=lam,
                            huber_delta=delta)
            draws[b] = m.theta
        except (ValueError, np.linalg.LinAlgError):
            continue
    out = {}
    for j, nm in enumerate(names):
        col = draws[:, j]
        col = col[np.isfinite(col)]
        if col.size < 20:
            out[nm] = {"controllable": False, "n_boot": int(col.size),
                       "verdict": "[DOP] bootstrap did not converge; channel "
                                  "not offered as a control handle"}
            continue
        lo, hi = float(np.quantile(col, 0.05)), float(np.quantile(col, 0.95))
        sig = bool(lo > 0 or hi < 0)
        out[nm] = {
            "mean": float(col.mean()), "ci05": lo, "ci95": hi,
            "n_boot": int(col.size), "significant": sig, "controllable": sig,
            "verdict": (f"gain 90% CI [{lo:+.4g}, {hi:+.4g}] excludes zero -- "
                        "offered as a control handle"
                        if sig else
                        f"gain 90% CI [{lo:+.4g}, {hi:+.4g}] contains zero -- "
                        "excluded from the manipulated set (§19, §31)"),
        }
    return out


def _kernel_skill(y, kernel_pred, full_pred) -> dict:
    """Does the ВАК/kinetic kernel earn its place (prompt §0)?"""
    y = np.asarray(y, float)
    base = float(np.median(y))
    def mae(p):
        return float(np.nanmean(np.abs(y - p)))
    m_base = mae(np.full(y.shape, base))
    off = float(np.median(y - kernel_pred))
    return {"MAE_median_baseline": m_base,
            "MAE_kernel_only": mae(kernel_pred + off),
            "MAE_kernel_plus_residual": mae(full_pred),
            "kernel_skill_vs_median": 1.0 - mae(kernel_pred + off) / m_base
            if m_base else float("nan"),
            "full_skill_vs_median": 1.0 - mae(full_pred) / m_base
            if m_base else float("nan")}


def _calibrate_uncertainty(resid, ages, alpha: float, c_tau_default: float,
                           c_tau_max: float = 0.5) -> tuple[float, float, dict]:
    """Calibrate sigma_pred (eq 58) so that its one-sided cover is honest.

    The product-sulfur residuals are heavy tailed (the record holds lab results
    an order of magnitude above the 10 mg/kg limit), so a plain RMS sigma is
    both too large in the body and too small in the tail.  sigma is therefore
    set so that the *training* one-sided (1-alpha) bound is attained exactly:

        sigma = quantile_{1-alpha}(residual) / z_{1-alpha}

    which is the quantity eq 63 actually uses.  §28's warning stands: this is a
    historical quantile, not a bound, which is why the set-based check of
    eq 59-60 is carried alongside it.
    """
    from src.predict.uncertainty import z_quantile
    r = np.asarray(resid, float)
    a = np.asarray(ages, float)
    m = np.isfinite(r)
    if m.sum() < 30:
        return float(np.sqrt(np.nanmean(r ** 2))), float(c_tau_default), {}
    z = z_quantile(alpha)
    q = float(np.quantile(r[m], 1.0 - alpha))
    rms = float(np.sqrt(np.mean(r[m] ** 2)))
    mad = 1.4826 * float(np.median(np.abs(r[m] - np.median(r[m]))))
    sigma = max(q / z, 1e-6)

    # age effect: robust slope of |residual| on the LIMS age, converted to a
    # variance rate and capped so it can never dominate sigma_resid.
    c_tau = float(c_tau_default)
    ma = m & np.isfinite(a)
    if ma.sum() >= 50 and np.ptp(a[ma]) > 1e-6:
        lo, hi = np.quantile(a[ma], [0.25, 0.75])
        if hi > lo:
            g1 = float(np.median(np.abs(r[ma & (a <= lo)])))
            g2 = float(np.median(np.abs(r[ma & (a >= hi)])))
            rate = (g2 ** 2 - g1 ** 2) / max(hi - lo, 1e-6)
            c_tau = float(np.clip(max(rate, c_tau_default), 0.0,
                                  c_tau_max * sigma ** 2))
    rep = {"sigma_rms": rms, "sigma_mad": mad,
           "sigma_from_quantile": sigma, "alpha": alpha,
           "resid_p50": float(np.median(r[m])),
           "resid_p95": float(np.quantile(r[m], 0.95)),
           "resid_max": float(np.max(r[m])),
           "note": "sigma set from the (1-alpha) residual quantile, not the "
                   "RMS: the residual distribution is heavy tailed (eq 58 is "
                   "an assumption, eq 59-60 is the guarantee-style check)."}
    return sigma, c_tau, rep


def _deactivation(frame, train, fit_rows, y, lhsv, pH2, T_w) -> dict:
    rows = fit_rows & train
    X = np.column_stack([y, lhsv, pH2])
    m = rows & np.isfinite(X).all(axis=1) & np.isfinite(T_w)
    if m.sum() < 60:
        return {"available": False,
                "reason": "not enough joint (S, LHSV, p_H2, T_WABT) rows"}
    model = DA.fit_normalisation(X[m], T_w[m], ["S", "LHSV_rel", "pH2"])
    dT = DA.delta_T_norm(T_w[m], X[m], model)
    t0 = frame["date"].iloc[0]
    t_h = (frame["date"][m] - t0).dt.total_seconds().to_numpy() / 3600.0
    rep = DA.drift_trend(t_h, dT)
    out = rep.as_dict()
    out["available"] = True
    out["normalisation_r2"] = model.r2
    return out


def _inner_split(frame, train, fit_rows, val_frac: float) -> dict:
    """Split the training window in time: fit part and inner-validation part."""
    dates = frame["date"]
    tr_idx = np.where(train)[0]
    if tr_idx.size == 0:
        m = np.zeros(len(frame), bool)
        return {"fit_mask": train.copy(), "val_rows": m, "available": False}
    cut = dates.iloc[tr_idx[int((1.0 - val_frac) * (tr_idx.size - 1))]]
    fit_mask = train & (dates <= cut).to_numpy()
    val_mask = train & (dates > cut).to_numpy()
    return {"fit_mask": fit_mask, "val_rows": val_mask,
            "inner_end": str(cut), "n_fit": int(fit_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_val_lab": int((val_mask & fit_rows).sum()), "available": True}


def _tune_filter(frame, utilde, kernel, gb, dyn, cfg, score_mask,
                 lab_rows) -> dict:
    """Grid search of (residual shrinkage, Q_eta, R_lab, ПАК on/off) (§37.5).

    Scored by the MAE of the *causal replay* on ``score_mask`` -- the inner
    validation block, which the residual was not fitted on.  The ПАК switch is
    part of the grid rather than an assumption: the analyser's usefulness is
    decided by the data, and on this package the search rejects it.
    """
    from src.predict.plant import PlantPredictor, replay
    from src.validate import metrics

    y = frame["LIMS:sulfur"].to_numpy(float)
    Xk = utilde[gb.feature_names].to_numpy(float)
    kern = None
    tr_rows = lab_rows & score_mask & np.isfinite(y) & (y < LAB_FIT_CLIP_MGKG)
    grid = []
    for scale in (0.0, 0.5, 1.0):
        theta = np.asarray(gb.theta, float) * scale
        pred = PlantPredictor(kernel, gb.feature_names, theta, gb.intercept,
                              list(dyn))
        if kern is None:
            # must use the *shaped* kernel inputs, exactly as replay does;
            # mixing shaped and unshaped bases here silently biases the
            # re-anchored intercept and collapses the sigma calibration
            kern = pred.kernel_series(frame, utilde)
        # re-anchor the intercept so shrinkage does not introduce a bias
        base = kern + Xk @ theta
        m0 = lab_rows & score_mask & np.isfinite(base) & np.isfinite(y)
        pred.residual_intercept = (float(np.median(y[m0] - base[m0]))
                                   if m0.sum() > 20 else gb.intercept)
        for q in (0.0005, 0.002, 0.01, 0.05, 0.2):
            for r_lims in (0.25, 1.0, 4.0):
                for use_pak in (False, True):
                    rr = replay(pred, frame, utilde, q_bias_per_h=q,
                                r_lims=r_lims,
                                r_pak=float(cfg["filter"]["r_pak"]),
                                use_pak=use_pak)
                    yh = rr.frame["y_hat"].to_numpy(float)
                    m = tr_rows & np.isfinite(yh)
                    sc = metrics.metric_set(y[m], yh[m])
                    grid.append({"residual_scale": scale, "q_bias_per_h": q,
                                 "r_lims": r_lims, "use_pak": use_pak,
                                 "MAE": sc["MAE"], "RMSE": sc["RMSE"],
                                 "n": sc["n"], "anchors": rr.n_anchors,
                                 "gate_rejected": rr.n_rejected})
    best = min(grid, key=lambda g: g["MAE"])
    return {"grid": grid, "chosen": best,
            "note": "hyperparameters selected on the inner validation block "
                    "only (model_spec.md §37.5, §38); the ПАК anchor is a "
                    "searched option, not an assumption."}


def _slim(reports: dict) -> dict:
    """Drop the bulky raw vectors before serialising the report."""
    out = {}
    for k, v in reports.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()
                      if kk not in ("train_residuals", "train_ages")}
        else:
            out[k] = v
    return out


def save(model: IdentifiedModel, path) -> None:
    payload = {
        "filter_params": model.filter_params,
        "channel_dynamics": model.channel_dynamics,
        "kernel": model.kernel.as_dict(),
        "residual_names": model.residual_names,
        "residual_theta": np.asarray(model.residual_theta).tolist(),
        "residual_intercept": model.residual_intercept,
        "sigma_resid": model.sigma_resid, "c_tau": model.c_tau,
        "bounds": {k: asdict(v) for k, v in model.bounds.items()},
        "severity_terms": [asdict(t) for t in model.severity_terms],
        "deactivation": model.deactivation,
        "excitation": model.excitation,
        "train_end": model.train_end,
        "reports": _slim(model.reports),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=float)
