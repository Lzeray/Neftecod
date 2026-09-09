"""FIR shaping kernels and regressors -- model_spec.md §15, eq (38).

    (38)  y_k = sum_j sum_l g_{j,l} u_{j,k-l} + nu_k,   K_j ~ sum_l g_{j,l}

Two distinct uses:
  * ``fir_weights`` -- a *smoothing* kernel used to build an effective input
    (non-negative, sums to 1, so it cannot change the steady-state gain);
  * ``fir_regressor_matrix`` -- the lagged design matrix used by the
    identification step (src/ident/fir_ident.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fir_weights(length: int, tau_samples: float, kind: str = "exp") -> np.ndarray:
    """Non-negative smoothing kernel of unit sum.

    ``exp``  -- discrete first-order kernel, matches eq 33 in steady state;
    ``box``  -- moving average.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if kind == "box":
        w = np.ones(length, dtype=float)
    elif kind == "exp":
        if tau_samples <= 0:
            w = np.zeros(length); w[0] = 1.0
        else:
            a = np.exp(-1.0 / tau_samples)
            w = a ** np.arange(length, dtype=float)
    else:
        raise ValueError(f"unknown kernel {kind!r}")
    total = w.sum()
    if total <= 0:
        raise ValueError("degenerate kernel")
    return w / total


def apply_fir(u, w) -> np.ndarray:
    """Causal convolution sum_l w_l u_{k-l}; the first len(w)-1 rows are NaN."""
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    L = len(w)
    n = len(u)
    out = np.full(n, np.nan)
    if n < L:
        return out
    # np.convolve gives the same result; done explicitly to keep the alignment
    # (out[k] uses u[k-L+1 .. k]) unambiguous.
    conv = np.convolve(u, w, mode="full")[: n]
    out[L - 1:] = conv[L - 1:]
    return out


def fir_regressor_matrix(u, length: int) -> np.ndarray:
    """Design matrix [u_k, u_{k-1}, ..., u_{k-L+1}], shape (n, L)."""
    from src.features.lag import pure_delay
    u = np.asarray(u, dtype=float)
    return np.column_stack([pure_delay(u, l) for l in range(length)])


def steady_state_gain_fir(g) -> float:
    """K_j ~ sum_l g_{j,l}  (eq 38)."""
    return float(np.sum(np.asarray(g, dtype=float)))
