"""Instrumental variables -- model_spec.md §18, eq (42)-(43).

    (42)  E[phi_k eps_k] = 0                       (needed for OLS consistency)
    (43)  E[z_k eps_k] = 0, rank E[z_k phi_k'] = p
          theta_IV = (Z' Phi)^{-1} Z' Y

The historical data were collected under closed-loop control, so u_k is
correlated with the noise and eq 42 fails: a naive regression of quality on u
is biased (§18).  Valid instruments are *external* set points and lagged
exogenous signals; an arbitrary lagged tag is not an instrument, which is why
:func:`iv_diagnostics` reports instrument relevance and the caller must supply
the instruments explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IVResult:
    theta: np.ndarray
    residuals: np.ndarray
    sigma2: float
    cov: np.ndarray
    relevance: float          # smallest singular value of Z'Phi / n
    cond: float


def iv_fit(Phi, Y, Z, ridge: float = 0.0, weights=None) -> IVResult:
    """Two-stage / generalised IV estimate of eq (43).

    ``Z`` may have more columns than ``Phi`` (over-identified), in which case
    the two-stage least-squares projection is used, which reduces exactly to
    eq 43 when ``Z`` is square and invertible.
    """
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n, p = Phi.shape
    if Z.shape[0] != n or Y.size != n:
        raise ValueError("Phi, Y and Z must share the same number of rows")
    if Z.shape[1] < p:
        raise ValueError("under-identified: need at least as many instruments "
                         "as regressors (eq 43 rank condition)")
    w = np.ones(n) if weights is None else np.asarray(weights, float).ravel()
    if w.size != n:
        raise ValueError("weight vector length mismatch")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    Zw = Z * w[:, None]
    # first stage: project the regressors on the instrument space
    ZtZ = Zw.T @ Z + ridge * np.eye(Z.shape[1])
    Pi = np.linalg.solve(ZtZ, Zw.T @ Phi)
    Phi_hat = Z @ Pi
    H = (Phi_hat * w[:, None]).T @ Phi + ridge * np.eye(p)
    theta = np.linalg.solve(H, (Phi_hat * w[:, None]).T @ Y)
    res = Y - Phi @ theta
    dof = max(n - p, 1)
    sigma2 = float(res @ res / dof)
    Hi = np.linalg.inv(H)
    cov = sigma2 * Hi @ ((Phi_hat * w[:, None]).T @ Phi_hat) @ Hi.T
    sv = np.linalg.svd(Z.T @ Phi / n, compute_uv=False)
    return IVResult(theta=theta, residuals=res, sigma2=sigma2, cov=cov,
                    relevance=float(sv.min()), cond=float(sv.max() / max(sv.min(), 1e-300)))


def iv_diagnostics(Phi, Z, min_relevance: float = 1e-6) -> dict:
    """Rank / relevance check of eq (43) before trusting an IV estimate."""
    Phi = np.atleast_2d(np.asarray(Phi, float))
    Z = np.atleast_2d(np.asarray(Z, float))
    M = Z.T @ Phi / max(Phi.shape[0], 1)
    sv = np.linalg.svd(M, compute_uv=False)
    return {
        "rank": int(np.linalg.matrix_rank(M)),
        "p": int(Phi.shape[1]),
        "min_singular_value": float(sv.min()) if sv.size else 0.0,
        "relevant": bool(sv.size and sv.min() >= min_relevance
                         and np.linalg.matrix_rank(M) >= Phi.shape[1]),
    }


def robust_iv_fit(Phi, Y, Z, delta: float = 1.345, ridge: float = 0.0,
                  weights=None, max_iter: int = 50, tol: float = 1e-10
                  ) -> IVResult:
    """IV (eq 43) made robust to outliers with the Huber weights of eq 41.

    Iteratively reweighted IV: each pass solves eq 43 with the Huber weight
    psi(r/sigma)/(r/sigma) attached to every row.  It keeps the consistency of
    IV under closed-loop data (§18) while suppressing the mis-decoded samples
    that §17 warns about.  Neither step alone is enough: Huber does not remove
    the closed-loop bias, and plain IV is not robust.
    """
    from src.ident.huber import huber_weight, mad_sigma

    Phi = np.atleast_2d(np.asarray(Phi, float))
    Y = np.asarray(Y, float).ravel()
    w0 = (np.ones(len(Y)) if weights is None
          else np.asarray(weights, float).ravel())
    res = iv_fit(Phi, Y, Z, ridge=ridge, weights=w0)
    theta = res.theta
    for _ in range(max_iter):
        r = Y - Phi @ theta
        sc = mad_sigma(r)
        w = huber_weight(r / sc, delta) * w0
        res = iv_fit(Phi, Y, Z, ridge=ridge, weights=w)
        if np.max(np.abs(res.theta - theta)) <= tol * (1 + np.max(np.abs(theta))):
            theta = res.theta
            break
        theta = res.theta
    return res
