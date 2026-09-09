"""Interval honesty and spec-violation rates -- model_spec.md §38.

Coverage of ``y_hat +- z_{1-alpha} sigma_pred`` on held-out LIMS results must
be close to ``1 - alpha``; that is the check on whether sigma_pred (eq 58) is
honest rather than decorative.  False and missed spec violations are reported
alongside it, because a well-calibrated interval that never flags a real
violation is still useless.
"""
from __future__ import annotations

import numpy as np

from src.predict.uncertainty import z_quantile


def interval_coverage(y_true, y_pred, sigma, alpha: float = 0.05) -> dict:
    """Empirical coverage of the two-sided (1-alpha) prediction interval."""
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    sg = np.asarray(sigma, float).ravel()
    if sg.size == 1:
        sg = np.full(yt.shape, float(sg[0]))
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(sg) & (sg > 0)
    if m.sum() == 0:
        return {"n": 0, "coverage": float("nan"), "target": 1 - alpha}
    z = z_quantile(alpha / 2.0)
    lo, hi = yp[m] - z * sg[m], yp[m] + z * sg[m]
    inside = (yt[m] >= lo) & (yt[m] <= hi)
    cov = float(inside.mean())
    return {"n": int(m.sum()), "coverage": cov, "target": 1.0 - alpha,
            "z": float(z), "mean_width": float(np.mean(hi - lo)),
            "honest": bool(abs(cov - (1 - alpha)) <= 0.10)}


def one_sided_coverage(y_true, y_pred, sigma, alpha: float = 0.05) -> dict:
    """Coverage of the *one-sided* bound used by the chance constraint (eq 63)."""
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    sg = np.asarray(sigma, float).ravel()
    if sg.size == 1:
        sg = np.full(yt.shape, float(sg[0]))
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(sg) & (sg > 0)
    if m.sum() == 0:
        return {"n": 0, "coverage": float("nan"), "target": 1 - alpha}
    z = z_quantile(alpha)
    below = yt[m] <= yp[m] + z * sg[m]
    return {"n": int(m.sum()), "coverage": float(below.mean()),
            "target": 1.0 - alpha, "z": float(z)}


def violation_rates(y_true, y_pred, limit: float, backoff=0.0) -> dict:
    """False / missed spec-violation rates against the hard limit."""
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    bo = np.asarray(backoff, float).ravel()
    if bo.size == 1:
        bo = np.full(yt.shape, float(bo[0]))
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return {"n": 0}
    actual = yt[m] > limit
    flagged = (yp[m] + bo[m]) > limit
    tp = int(np.sum(actual & flagged)); fp = int(np.sum(~actual & flagged))
    fn = int(np.sum(actual & ~flagged)); tn = int(np.sum(~actual & ~flagged))
    return {"n": int(m.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "missed_rate": fn / max(tp + fn, 1),
            "false_alarm_rate": fp / max(fp + tn, 1),
            "actual_violations": int(actual.sum())}
