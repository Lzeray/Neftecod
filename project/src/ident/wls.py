"""Weighted, regularised least squares -- model_spec.md §16, eq (39)-(40).

    (39)  theta_hat = argmin (Y - Phi theta)^T W (Y - Phi theta)
                             + lambda ||L(theta - theta_0)||^2
    (40)  theta_hat = (Phi^T W Phi + lambda L^T L)^{-1}
                      (Phi^T W Y + lambda L^T L theta_0)
          grad^2 J = 2(Phi^T W Phi + lambda L^T L) >= 0

W carries source priority and freshness; theta_0 anchors the fit to the physical
kernel / the ВАК regressions (§16).  The problem is convex only for *fixed*
Phi, W, L -- delays and kinetic parameters are fitted in the outer loop (§37).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WLSResult:
    theta: np.ndarray
    residuals: np.ndarray
    sigma2: float
    cov: np.ndarray
    hessian: np.ndarray
    dof: int

    def predict(self, Phi):
        return np.asarray(Phi, dtype=float) @ self.theta


def _prep(Phi, Y, w, L, theta0):
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    Y = np.asarray(Y, dtype=float).ravel()
    n, p = Phi.shape
    if Y.size != n:
        raise ValueError(f"Y has {Y.size} rows, Phi has {n}")
    if n == 0 or p == 0:
        raise ValueError("empty regression problem")
    w = np.ones(n) if w is None else np.asarray(w, dtype=float).ravel()
    if w.size != n:
        raise ValueError("weight vector length mismatch")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    L = np.eye(p) if L is None else np.atleast_2d(np.asarray(L, dtype=float))
    if L.shape[1] != p:
        raise ValueError("L must have p columns")
    theta0 = np.zeros(p) if theta0 is None else np.asarray(theta0, float).ravel()
    if theta0.size != p:
        raise ValueError("theta0 length mismatch")
    return Phi, Y, w, L, theta0


def wls(Phi, Y, weights=None, lam: float = 0.0, L=None, theta0=None) -> WLSResult:
    """Solve eq (39) in the closed form of eq (40)."""
    Phi, Y, w, L, theta0 = _prep(Phi, Y, weights, L, theta0)
    n, p = Phi.shape
    PW = Phi * w[:, None]
    H = PW.T @ Phi + lam * (L.T @ L)
    rhs = PW.T @ Y + lam * (L.T @ L) @ theta0
    theta = np.linalg.solve(H, rhs)
    res = Y - Phi @ theta
    dof = max(n - p, 1)
    sigma2 = float((w * res ** 2).sum() / dof) if w.sum() > 0 else float("nan")
    Hi = np.linalg.inv(H)
    cov = sigma2 * Hi @ (PW.T @ Phi) @ Hi
    return WLSResult(theta=theta, residuals=res, sigma2=sigma2, cov=cov,
                     hessian=2.0 * H, dof=dof)


def hessian(Phi, weights=None, lam: float = 0.0, L=None) -> np.ndarray:
    """grad^2 J = 2(Phi^T W Phi + lambda L^T L)   (eq 40); always PSD."""
    Phi, _, w, L, _ = _prep(Phi, np.zeros(np.atleast_2d(Phi).shape[0]),
                            weights, L, None)
    return 2.0 * ((Phi * w[:, None]).T @ Phi + lam * (L.T @ L))


def freshness_weights(age_h, tau_half_h: float = 24.0) -> np.ndarray:
    """W entries decaying with measurement age (§16: "W carries freshness").

    ``w = 2^{-age/tau_half}``, capped at 1 and floored at a small positive
    value so a stale sample is down-weighted rather than silently dropped.
    """
    a = np.asarray(age_h, dtype=float)
    w = np.power(2.0, -np.clip(a, 0.0, None) / float(tau_half_h))
    return np.where(np.isfinite(w), np.clip(w, 1e-6, 1.0), 1e-6)
