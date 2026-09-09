"""Model stability -- model_spec.md §20, eq (45).

    (45)  discrete: rho(A) = max_i |lambda_i(A)| < 1
          continuous: Re lambda_i(A_c) < 0

§20 also notes that constraining rho(A(theta)) < 1 jointly with an arbitrary
parameterisation is *not* convex, so identification here checks stability after
the fact and projects rather than constraining inside the convex fit.
"""
from __future__ import annotations

import numpy as np


def spectral_radius(A) -> float:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_stable_discrete(A, tol: float = 0.0) -> bool:
    """rho(A) < 1 - tol   (eq 45)."""
    return spectral_radius(A) < 1.0 - tol


def is_stable_continuous(Ac, tol: float = 0.0) -> bool:
    Ac = np.atleast_2d(np.asarray(Ac, dtype=float))
    if Ac.size == 0:
        return True
    return bool(np.all(np.real(np.linalg.eigvals(Ac)) < -tol))


def arx_companion(a: np.ndarray) -> np.ndarray:
    """Companion matrix of the AR polynomial y_k = sum_i a_i y_{k-i} (eq 35)."""
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    if n == 0:
        return np.zeros((0, 0))
    A = np.zeros((n, n))
    A[0, :] = a
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    return A


def arx_is_stable(a: np.ndarray, tol: float = 0.0) -> bool:
    return is_stable_discrete(arx_companion(a), tol)


def project_to_stable(a: np.ndarray, rho_max: float = 0.98,
                      max_iter: int = 200) -> np.ndarray:
    """Shrink the AR coefficients until rho(companion) <= rho_max.

    ``a_i <- gamma^i a_i`` maps the poles ``z_j -> gamma z_j``, so a bisection on
    gamma reaches any target radius.  Used only when the unconstrained convex
    fit of §16 returns an unstable polynomial (§20).
    """
    a = np.asarray(a, dtype=float).ravel()
    if arx_is_stable(a) and spectral_radius(arx_companion(a)) <= rho_max:
        return a.copy()
    lo, hi = 0.0, 1.0
    best = np.zeros_like(a)
    for _ in range(max_iter):
        g = 0.5 * (lo + hi)
        cand = a * g ** np.arange(1, a.size + 1)
        if spectral_radius(arx_companion(cand)) <= rho_max:
            best, lo = cand, g
        else:
            hi = g
        if hi - lo < 1e-12:
            break
    return best
