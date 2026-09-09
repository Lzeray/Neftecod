"""Forecast metrics -- model_spec.md §38, eq (67)-(69).

    (67)  e_{k,h} = y_{k+h} - y_hat_{k+h|k}
    (68)  MAE_h, RMSE_h, Bias_h
    (69)  persistence baseline y_hat = y_k

One-step, multi-step and step-response reproduction are reported separately
and never conflated (§38 closing note).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def errors(y_true, y_pred) -> np.ndarray:
    """e = y_true - y_pred  (eq 67)."""
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    if yt.size != yp.size:
        raise ValueError("length mismatch")
    return yt - yp


def mae(y_true, y_pred) -> float:
    e = errors(y_true, y_pred)
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def rmse(y_true, y_pred) -> float:
    e = errors(y_true, y_pred)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def bias(y_true, y_pred) -> float:
    e = errors(y_true, y_pred)
    e = e[np.isfinite(e)]
    return float(np.mean(e)) if e.size else float("nan")


def metric_set(y_true, y_pred) -> dict:
    """MAE/RMSE/Bias with the standard error of the MAE.

    Without an error bar a difference of a few hundredths of a mg/kg reads as
    an improvement when it is noise.  On this unit the MAE standard error is
    ~0.28 mg/kg on 468 held-out results, which is larger than most of the
    differences between candidate models -- so the error bar is not optional
    decoration, it is what stops the report from claiming skill it cannot
    demonstrate.
    """
    e = errors(y_true, y_pred)
    e = e[np.isfinite(e)]
    n = int(e.size)
    se = (float(np.std(np.abs(e), ddof=1) / np.sqrt(n)) if n > 1
          else float("nan"))
    return {"n": n, "MAE": mae(y_true, y_pred), "MAE_se": se,
            "RMSE": rmse(y_true, y_pred), "Bias": bias(y_true, y_pred)}


def paired_mae_difference(y_true, y_pred_a, y_pred_b) -> dict:
    """Paired comparison of two predictors on the same rows.

    Returns the mean of ``|e_a| - |e_b|`` with its standard error, which is the
    only defensible way to say whether A is better than B on a shared sample.
    """
    ea = np.abs(errors(y_true, y_pred_a))
    eb = np.abs(errors(y_true, y_pred_b))
    m = np.isfinite(ea) & np.isfinite(eb)
    if m.sum() < 2:
        return {"n": int(m.sum()), "diff": float("nan"), "se": float("nan"),
                "significant": False}
    d = ea[m] - eb[m]
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mean = float(d.mean())
    return {"n": int(d.size), "diff_MAE": mean, "se": se,
            "t": (mean / se if se > 0 else float("nan")),
            "significant": bool(se > 0 and abs(mean) > 1.96 * se),
            "note": "negative means the first predictor has the lower MAE"}


def persistence(y, h: int) -> np.ndarray:
    """y_hat_{k+h|k} = y_k  (eq 69), aligned with y_{k+h}."""
    y = np.asarray(y, float).ravel()
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if h >= y.size:
        return np.full(0, np.nan)
    return y[:-h]


def horizon_table(y, preds: dict[int, np.ndarray]) -> pd.DataFrame:
    """MAE/RMSE/Bias per horizon plus the persistence baseline (eq 68-69)."""
    y = np.asarray(y, float).ravel()
    rows = []
    for h in sorted(preds):
        yp = np.asarray(preds[h], float).ravel()
        yt = y[h:h + yp.size]
        m = metric_set(yt, yp[:yt.size])
        base = persistence(y, h)
        mb = metric_set(y[h:h + base.size], base)
        rows.append({"h": h, **{k: m[k] for k in ("n", "MAE", "RMSE", "Bias")},
                     "MAE_persist": mb["MAE"], "RMSE_persist": mb["RMSE"],
                     "skill_MAE": (1.0 - m["MAE"] / mb["MAE"])
                     if mb["MAE"] and np.isfinite(mb["MAE"]) else np.nan})
    return pd.DataFrame(rows)


def step_response_error(y_true_step, y_pred_step) -> dict:
    """Reproduction of the response to an input change (§38, third category)."""
    yt = np.asarray(y_true_step, float).ravel()
    yp = np.asarray(y_pred_step, float).ravel()
    m = metric_set(yt, yp)
    gain_t = float(yt[-1] - yt[0]) if yt.size > 1 else np.nan
    gain_p = float(yp[-1] - yp[0]) if yp.size > 1 else np.nan
    m["gain_true"] = gain_t
    m["gain_pred"] = gain_p
    m["gain_rel_err"] = (abs(gain_p - gain_t) / abs(gain_t)
                         if gain_t not in (0.0, np.nan) and np.isfinite(gain_t)
                         else np.nan)
    return m
