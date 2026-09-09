"""Chance constraints -- model_spec.md §32, eq (63); §28 eq (60).

    (63)  Pr(S_prod <= 10) >= 1 - alpha  <=>  S_hat + z_{1-alpha} sigma_pred <= 10
          conservative check:             S_hat + eps_S^+ <= 10

The z sigma term is an *automatic* back-off: as the data get older or the
operating point moves out of the identified domain, sigma_pred (eq 58) grows
and the recommendation becomes more conservative on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.predict.uncertainty import z_quantile


@dataclass
class ChanceCheck:
    y_hat: float
    sigma: float
    z: float
    limit: float
    backoff: float
    value: float          # y_hat + backoff
    satisfied: bool
    margin: float
    p_violation: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def backoff(sigma: float, alpha: float) -> float:
    """z_{1-alpha} * sigma_pred  (eq 63)."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return z_quantile(alpha) * float(sigma)


def check_max(y_hat: float, sigma: float, limit: float, alpha: float
              ) -> ChanceCheck:
    """Deterministic equivalent of Pr(y <= limit) >= 1-alpha  (eq 63)."""
    from scipy.stats import norm
    z = z_quantile(alpha)
    bo = z * float(sigma)
    val = float(y_hat) + bo
    p = float(norm.sf((limit - y_hat) / sigma)) if sigma > 0 else \
        float(y_hat > limit)
    return ChanceCheck(float(y_hat), float(sigma), z, float(limit), bo, val,
                       val <= limit + 1e-12, float(limit) - val, p)


def check_min(y_hat: float, sigma: float, limit: float, alpha: float
              ) -> ChanceCheck:
    """Pr(y >= limit) >= 1-alpha  ->  y_hat - z sigma >= limit."""
    from scipy.stats import norm
    z = z_quantile(alpha)
    bo = z * float(sigma)
    val = float(y_hat) - bo
    p = float(norm.cdf((limit - y_hat) / sigma)) if sigma > 0 else \
        float(y_hat < limit)
    return ChanceCheck(float(y_hat), float(sigma), z, float(limit), bo, val,
                       val >= limit - 1e-12, val - float(limit), p)


def set_based_check(y_hat: float, eps_plus: float, limit: float) -> dict:
    """Conservative set-based check of eq (60): y_hat + eps^+ <= limit.

    §28: this is the guarantee-style check that accompanies the Gaussian one,
    because a historical error quantile is not a bound on the disturbance set.
    """
    val = float(y_hat) + float(eps_plus)
    return {"value": val, "limit": float(limit), "eps_plus": float(eps_plus),
            "satisfied": val <= float(limit) + 1e-12,
            "margin": float(limit) - val}


def saa_violation_rate(y_samples, limit: float) -> float:
    """Sample-average approximation of Pr(y > limit) (§35)."""
    y = np.asarray(y_samples, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float((y > limit).mean())
