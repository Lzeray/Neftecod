"""Prediction uncertainty -- model_spec.md §28, eq (58)-(60).

    (58)  sigma_pred^2 = sigma_resid^2 + P_eta(t) + c_tau tau(t) + sigma_lev^2(x)
    (59)  E_{h+1} = A E_h (+) W,  Y^err_h = C E_h (+) V
          eps^+_{j,h} = sup_{e in E_h, v in V} (c_j' e + v_j)
    (60)  y_{j,k+h} <= y_hat + eps^+_{j,h}

§28 is explicit that the Gaussian back-off z sigma is an *assumption*, not a
guarantee: the historical error quantile is not a bound on W.  Both estimates
are therefore carried for the hard sulfur limit -- eq 58 as the working number
and eq 60 as the conservative check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def z_quantile(alpha: float) -> float:
    """z_{1-alpha} of eq 63."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    return float(norm.ppf(1.0 - alpha))


# ----------------------------------------------------------------- eq 58
@dataclass
class UncertaintyModel:
    """Stochastic prediction variance of eq (58).

    The extrapolation term needs care.  Two different errors grow away from the
    training domain and only one of them has a known form:

      * *parameter* uncertainty -- the textbook prediction variance of a linear
        model is sigma^2 (1 + x'(X'X)^-1 x), whose leverage is the Mahalanobis
        distance divided by the number of training points.  With ~1e3 lab
        results that term is negligible, which is correct: the level is very
        well determined.
      * *structural* error -- how wrong the kernel becomes on a feed it has
        never seen.  Nothing in the package determines its shape.

    Inflating sigma by the raw Mahalanobis distance (as if the second error
    scaled like the first) turns one unusual feed sample into a 4x wider
    interval and makes every recommendation impossible.  So the term is kept,
    because §28 asks uncertainty to grow on unusual data, but it is **capped**
    at ``lev_cap`` (a declared [DOP] choice) and the raw distance is reported
    separately as an explicit extrapolation flag, which is the honest signal:
    "this operating point is outside where the model was identified".
    """
    sigma_resid: float                 # residual sd of the fitted predictor
    c_tau: float = 0.05                # variance growth per hour of data age
    lev_scale: float = 1.0             # multiplier on the leverage term
    lev_cap: float = 3.0               # [DOP] max variance multiple from leverage
    horizon_growth: float = 0.0        # extra variance per prediction step
    x_mean: np.ndarray | None = None   # training-domain centre
    x_cov_inv: np.ndarray | None = None  # inverse covariance for the leverage
    n_train: int = 0                   # training points behind the domain

    def mahalanobis2(self, x) -> float:
        """Squared Mahalanobis distance from the training-domain centre."""
        if self.x_mean is None or self.x_cov_inv is None:
            return 0.0
        v = np.asarray(x, float).ravel() - self.x_mean
        if v.size != self.x_mean.size:
            raise ValueError("state dimension mismatch in the leverage term")
        return float(max(v @ self.x_cov_inv @ v, 0.0))

    def leverage(self, x) -> float:
        """Capped extrapolation multiplier sigma_lev^2(x)/sigma_resid^2."""
        if self.x_mean is None or self.x_cov_inv is None:
            return 0.0
        d2 = self.mahalanobis2(x)
        p = max(self.x_mean.size, 1)
        raw = self.lev_scale * d2 / p
        return float(min(raw, self.lev_cap))

    def extrapolation(self, x, threshold: float | None = None) -> dict:
        """Explicit out-of-domain report for the operator."""
        if self.x_mean is None or self.x_cov_inv is None:
            return {"available": False}
        d2 = self.mahalanobis2(x)
        p = max(self.x_mean.size, 1)
        thr = float(threshold) if threshold is not None else 4.0 * p
        v = np.asarray(x, float).ravel() - self.x_mean
        sd = np.sqrt(np.diag(np.linalg.inv(self.x_cov_inv)))
        z = v / np.where(sd > 0, sd, 1.0)
        return {"available": True, "mahalanobis2": d2, "dim": p,
                "threshold": thr, "outside": bool(d2 > thr),
                "per_dimension_z": z.tolist(),
                "capped": bool(self.lev_scale * d2 / p > self.lev_cap),
                "note": "distance of the operating point from the domain the "
                        "kernel was identified on; the variance contribution "
                        f"is capped at {self.lev_cap:g} x sigma_resid^2"}

    def sigma_pred(self, tau_h: float = 0.0, P_eta: float = 0.0, x=None,
                   h: int = 0) -> float:
        """sigma_pred of eq (58); monotone in tau, P_eta, leverage and h."""
        if tau_h < 0 or P_eta < 0 or h < 0:
            raise ValueError("age, offset variance and horizon must be >= 0")
        var = (self.sigma_resid ** 2 + float(P_eta)
               + self.c_tau * float(tau_h) + self.horizon_growth * float(h))
        if x is not None:
            var += self.leverage(x) * self.sigma_resid ** 2
        return float(np.sqrt(max(var, 0.0)))


def fit_domain(X) -> tuple[np.ndarray, np.ndarray]:
    """Training-domain centre and inverse covariance for the leverage term."""
    X = np.atleast_2d(np.asarray(X, float))
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    S = S + 1e-9 * np.eye(S.shape[0]) * max(np.trace(S) / S.shape[0], 1.0)
    return mu, np.linalg.inv(S)


# ----------------------------------------------------------------- eq 59-60
def set_error_bound(A, C, w_half, v_half, H: int, e0_half=None) -> np.ndarray:
    """eps^+_{j,h} for h = 1..H  (eq 59-60), exact for box W and V.

    With ``W = {w : |w_i| <= w_half_i}`` and ``E_0 = {e : |e_i| <= e0_half_i}``
    the reachable set stays a zonotope and its support function in direction
    ``c_j`` is the sum of the absolute row sums:

        eps^+_{j,h} = |c_j' A^h| e0_half + sum_{i=0}^{h-1} |c_j' A^i| w_half
                      + v_half_j
    """
    A = np.atleast_2d(np.asarray(A, float))
    C = np.atleast_2d(np.asarray(C, float))
    w = np.asarray(w_half, float).ravel()
    v = np.asarray(v_half, float).ravel()
    n, ny = A.shape[0], C.shape[0]
    if w.size != n:
        raise ValueError("w_half must have one entry per state")
    if v.size != ny:
        raise ValueError("v_half must have one entry per output")
    if np.any(w < 0) or np.any(v < 0):
        raise ValueError("set half-widths must be non-negative")
    e0 = np.zeros(n) if e0_half is None else np.asarray(e0_half, float).ravel()

    out = np.zeros((H, ny))
    for j in range(ny):
        cj = C[j]
        acc = np.abs(cj @ np.linalg.matrix_power(A, 0)) @ np.zeros(n)
        total = 0.0
        for h in range(1, H + 1):
            total += float(np.abs(cj @ np.linalg.matrix_power(A, h - 1)) @ w)
            init = float(np.abs(cj @ np.linalg.matrix_power(A, h)) @ e0)
            out[h - 1, j] = total + init + v[j]
        del acc
    return out


def empirical_half_widths(residuals, coverage: float = 1.0) -> float:
    """Half-width of the smallest symmetric interval covering ``coverage``.

    Used to *calibrate* W and V from data while remembering §28's warning: the
    historical quantile is an estimate of the disturbance set, not a proof.
    """
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    if not (0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")
    return float(np.quantile(np.abs(r), coverage))
