"""Continuous -> discrete conversion -- model_spec.md §11, eq (30)-(31).

    (31)  A = exp(A_c Ts),  B = int_0^Ts exp(A_c tau) dtau B_c,   likewise E
          with A = A_c^{-1}(A - I) B_c when A_c is invertible.

Implemented through the block matrix exponential (Van Loan), which is exact and
does not require A_c to be invertible.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def zoh(Ac: np.ndarray, Bc: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation, eq (31)."""
    Ac = np.atleast_2d(np.asarray(Ac, dtype=float))
    Bc = np.asarray(Bc, dtype=float)
    if Bc.ndim == 1:
        Bc = Bc.reshape(-1, 1)
    n, m = Ac.shape[0], Bc.shape[1]
    if Ac.shape[0] != Ac.shape[1]:
        raise ValueError("Ac must be square")
    if Bc.shape[0] != n:
        raise ValueError("Bc row count must match Ac")
    if Ts <= 0:
        raise ValueError("Ts must be positive")
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    E = expm(M * Ts)
    return E[:n, :n].copy(), E[:n, n:].copy()


def zoh_full(Ac, Bc, Ec, Ts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretise the (A_c, B_c, E_c) triple of eq 27 in one pass."""
    Bc = np.atleast_2d(np.asarray(Bc, float))
    Ec = np.atleast_2d(np.asarray(Ec, float))
    if Bc.shape[0] != np.atleast_2d(Ac).shape[0]:
        Bc = Bc.T
    if Ec.shape[0] != np.atleast_2d(Ac).shape[0]:
        Ec = Ec.T
    BE = np.hstack([Bc, Ec])
    A, BEd = zoh(Ac, BE, Ts)
    return A, BEd[:, :Bc.shape[1]].copy(), BEd[:, Bc.shape[1]:].copy()


def zoh_invertible(Ac, Bc, Ts) -> tuple[np.ndarray, np.ndarray]:
    """The closed form B = A_c^{-1}(A - I) B_c stated under eq 31.

    Only valid for an invertible A_c; used as an independent cross-check of
    :func:`zoh`, never as the production path.
    """
    Ac = np.atleast_2d(np.asarray(Ac, float))
    Bc = np.asarray(Bc, float)
    if Bc.ndim == 1:
        Bc = Bc.reshape(-1, 1)
    A = expm(Ac * Ts)
    B = np.linalg.solve(Ac, (A - np.eye(Ac.shape[0])) @ Bc)
    return A, B
