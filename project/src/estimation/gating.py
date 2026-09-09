"""Measurement admission rules -- model_spec.md §23.

Combines the chi^2 innovation gate (eq 49) with the staleness / zero-variance
test of §23 and a physical-range check.  A measurement must pass *all three*
before it is allowed to move the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimation.kalman import chi2_threshold


@dataclass
class GateDecision:
    accept: bool
    reason: str
    nis: float = float("nan")
    threshold: float = float("nan")


def range_check(value: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(value) and lo <= value <= hi)


def gate_scalar(y: float, y_pred: float, var_pred: float, R: float,
                stale: bool = False, valid_range: tuple[float, float] | None = None,
                gate_p: float = 0.99) -> GateDecision:
    """Scalar version of the §23 admission rule."""
    if not np.isfinite(y):
        return GateDecision(False, "no measurement")
    if stale:
        return GateDecision(False, "source flagged stale (zero variance window)")
    if valid_range is not None and not range_check(y, *valid_range):
        return GateDecision(False,
                            f"outside the physical range {valid_range}")
    S = float(var_pred) + float(R)
    if S <= 0:
        return GateDecision(False, "degenerate innovation covariance")
    nis = float((y - y_pred) ** 2 / S)
    thr = chi2_threshold(1, gate_p)
    if nis > thr:
        return GateDecision(False, f"innovation gate: NIS {nis:.2f} > {thr:.2f}",
                            nis, thr)
    return GateDecision(True, "accepted", nis, thr)
