"""Regularised FIR identification -- model_spec.md §15-16, eq (38)-(40).

The FIR parameterisation is an alternative to ARX (§15).  For smooth impulse
responses a regularised FIR (a Gaussian-process prior on g_{j,l}) is the
preferred form; the prior enters through the ``L`` matrix of eq 39, taken here
as a first-difference smoothness penalty, which is the discrete analogue of a
"the response is smooth" statement.
"""
from __future__ import annotations

import numpy as np

from src.features.fir import fir_regressor_matrix
from src.ident.wls import WLSResult, wls


def smoothness_matrix(length: int, n_channels: int = 1) -> np.ndarray:
    """Block-diagonal first-difference operator used as ``L`` in eq 39."""
    if length < 1:
        raise ValueError("length must be >= 1")
    D = np.zeros((length, length))
    for i in range(1, length):
        D[i, i - 1] = -1.0
        D[i, i] = 1.0
    D[0, 0] = 1.0
    if n_channels == 1:
        return D
    from scipy.linalg import block_diag
    return block_diag(*([D] * n_channels))


def fit_fir(u: dict[str, np.ndarray], y, length: int, lam: float = 1.0,
            weights=None) -> tuple[WLSResult, list[str]]:
    """Regularised FIR fit of eq (38) through eq (39)-(40)."""
    y = np.asarray(y, dtype=float)
    blocks, names = [], []
    for name, arr in u.items():
        blocks.append(fir_regressor_matrix(arr, length))
        names += [f"g[{name},{l}]" for l in range(length)]
    Phi = np.hstack(blocks)
    valid = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
    L = smoothness_matrix(length, len(u))
    w = None if weights is None else np.asarray(weights, float)[valid]
    res = wls(Phi[valid], y[valid], weights=w, lam=lam, L=L)
    return res, names


def impulse_response(theta: np.ndarray, names: list[str], channel: str
                     ) -> np.ndarray:
    return np.array([theta[i] for i, nm in enumerate(names)
                     if nm.startswith(f"g[{channel},")], dtype=float)
