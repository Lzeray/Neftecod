"""Robust regression -- model_spec.md §17, eq (41).

    (41)  rho_delta(r) = r^2/2                for |r| <= delta
                         delta|r| - delta^2/2 otherwise
          theta_hat = argmin sum_k w_k rho_delta((y_k - phi_k' theta)/sigma)
                             + lambda ||L(theta - theta_0)||^2

Convex for sigma > 0, w_k >= 0.  §17: it suppresses outliers but does not fix a
mis-decoded tag and does not remove errors-in-variables bias -- that is what
the IV step of §18 is for.
"""
from __future__ import annotations

import numpy as np

from src.ident.wls import WLSResult, _prep


def huber_rho(r, delta: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * a - 0.5 * delta ** 2)


def huber_psi(r, delta: float) -> np.ndarray:
    """rho' -- the influence function; bounded by +-delta."""
    r = np.asarray(r, dtype=float)
    return np.clip(r, -delta, delta)


def huber_weight(r, delta: float) -> np.ndarray:
    """IRLS weight psi(r)/r, defined as 1 at r = 0."""
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-300))


def mad_sigma(r) -> float:
    """Robust scale estimate 1.4826 * MAD, with a positive floor."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1.0
    s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    if s <= 0:
        s = float(np.std(r))
    return s if s > 0 else 1.0


def huber_fit(Phi, Y, delta: float = 1.345, weights=None, lam: float = 0.0,
              L=None, theta0=None, max_iter: int = 100, tol: float = 1e-10,
              sigma: float | None = None) -> WLSResult:
    """IRLS solution of eq (41).

    delta -> inf reduces exactly to the weighted least squares of eq 39/40.
    """
    Phi, Y, w0, L, th0 = _prep(Phi, Y, weights, L, theta0)
    n, p = Phi.shape
    LtL = L.T @ L
    theta = np.linalg.solve(Phi.T @ (Phi * w0[:, None]) + lam * LtL + 1e-12 * np.eye(p),
                            (Phi * w0[:, None]).T @ Y + lam * LtL @ th0)
    s = sigma
    for _ in range(max_iter):
        res = Y - Phi @ theta
        sc = mad_sigma(res) if s is None else float(s)
        wr = huber_weight(res / sc, delta) * w0
        PW = Phi * wr[:, None]
        H = PW.T @ Phi + lam * LtL
        new = np.linalg.solve(H, PW.T @ Y + lam * LtL @ th0)
        if np.max(np.abs(new - theta)) <= tol * (1.0 + np.max(np.abs(theta))):
            theta = new
            break
        theta = new
    res = Y - Phi @ theta
    sc = mad_sigma(res) if s is None else float(s)
    wr = huber_weight(res / sc, delta) * w0
    PW = Phi * wr[:, None]
    H = PW.T @ Phi + lam * LtL
    dof = max(n - p, 1)
    sigma2 = float((wr * res ** 2).sum() / dof)
    Hi = np.linalg.inv(H)
    return WLSResult(theta=theta, residuals=res, sigma2=sigma2,
                     cov=sigma2 * Hi @ (PW.T @ Phi) @ Hi, hessian=2.0 * H,
                     dof=dof)
