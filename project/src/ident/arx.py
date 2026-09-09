"""ARX / FIR identification -- model_spec.md §14-15, eq (35)-(38).

    (35)  y_k = sum_i a_i y_{k-i} + sum_j sum_l b_{j,l} u_{j,k-n_j-l}
                + sum_r sum_l e_{r,l} d_{r,k-l} + eps_k
    (36)  y_k = phi_k' theta + eps_k
    (37)  K_j = (sum_l b_{j,l}) / (1 - sum_i a_i)
    (38)  FIR: y_k = sum_j sum_l g_{j,l} u_{j,k-l},  K_j ~ sum_l g_{j,l}

An individual b_{j,l} is *not* the effect of the input; only eq 37 is (§14).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.lag import pure_delay


@dataclass
class ArxSpec:
    na: int
    inputs: dict[str, dict]      # name -> {"nb": int, "delay": int}
    disturbances: dict[str, dict] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return (self.na + sum(s["nb"] for s in self.inputs.values())
                + sum(s["nb"] for s in self.disturbances.values()))


def build_arx_regressors(y, u: dict[str, np.ndarray], spec: ArxSpec,
                         d: dict[str, np.ndarray] | None = None
                         ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Assemble Phi, Y and the parameter names of eq (35)-(36).

    Returns ``(Phi, Y, names, valid_mask)``; rows containing NaN in any used
    lag are excluded through ``valid_mask`` -- never imputed, because an
    imputed regressor would fabricate excitation.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    cols, names = [], []
    for i in range(1, spec.na + 1):
        cols.append(pure_delay(y, i))
        names.append(f"a{i}")
    for name, s in spec.inputs.items():
        uj = np.asarray(u[name], dtype=float)
        if uj.size != n:
            raise ValueError(f"input {name} length {uj.size} != {n}")
        for l in range(s["nb"]):
            cols.append(pure_delay(uj, int(s.get("delay", 0)) + l))
            names.append(f"b[{name},{l}]")
    d = d or {}
    for name, s in spec.disturbances.items():
        dr = np.asarray(d[name], dtype=float)
        if dr.size != n:
            raise ValueError(f"disturbance {name} length {dr.size} != {n}")
        for l in range(s["nb"]):
            cols.append(pure_delay(dr, int(s.get("delay", 0)) + l))
            names.append(f"e[{name},{l}]")
    Phi = np.column_stack(cols) if cols else np.empty((n, 0))
    valid = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
    return Phi, y, names, valid


def steady_state_gain(theta: np.ndarray, names: list[str], channel: str,
                      na: int) -> float:
    """K_j = sum_l b_{j,l} / (1 - sum_i a_i)   (eq 37).

    Returns NaN when the AR polynomial has unit gain (an integrator), where the
    static gain is not defined.
    """
    a_sum = float(sum(theta[i] for i, nm in enumerate(names)
                      if nm.startswith("a")))
    b_sum = float(sum(theta[i] for i, nm in enumerate(names)
                      if nm.startswith(f"b[{channel},")
                      or nm.startswith(f"e[{channel},")))
    den = 1.0 - a_sum
    if abs(den) < 1e-12:
        return float("nan")
    return b_sum / den


def ar_coefficients(theta: np.ndarray, names: list[str]) -> np.ndarray:
    return np.array([theta[i] for i, nm in enumerate(names)
                     if nm.startswith("a")], dtype=float)


def simulate_arx(theta: np.ndarray, names: list[str], spec: ArxSpec,
                 u: dict[str, np.ndarray], d: dict[str, np.ndarray] | None = None,
                 n: int | None = None, y0: float = 0.0) -> np.ndarray:
    """Free-run (multi-step) simulation of eq 35 -- no measured y is fed back."""
    any_u = next(iter(u.values()))
    n = int(n or np.asarray(any_u).size)
    a = ar_coefficients(theta, names)
    y = np.full(n, float(y0))
    idx = {nm: i for i, nm in enumerate(names)}
    d = d or {}
    for k in range(n):
        acc = 0.0
        for i, ai in enumerate(a, start=1):
            acc += ai * (y[k - i] if k - i >= 0 else y0)
        for src in (spec.inputs, spec.disturbances):
            book = u if src is spec.inputs else d
            pre = "b" if src is spec.inputs else "e"
            for name, s in src.items():
                arr = np.asarray(book[name], dtype=float)
                for l in range(s["nb"]):
                    kk = k - int(s.get("delay", 0)) - l
                    val = arr[kk] if kk >= 0 else 0.0
                    if np.isfinite(val):
                        acc += theta[idx[f"{pre}[{name},{l}]"]] * val
        y[k] = acc
    return y


def cross_correlation_delay(u, y, max_lag: int, min_lag: int = 0
                            ) -> tuple[int, np.ndarray]:
    """Delay candidate from the cross-correlation peak (§37 step 1).

    model_spec.md §12: the result is a *candidate to be tested*, never an
    established transport delay.
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    corrs = np.full(max_lag - min_lag + 1, np.nan)
    for i, lag in enumerate(range(min_lag, max_lag + 1)):
        ud = pure_delay(u, lag)
        m = np.isfinite(ud) & np.isfinite(y)
        if m.sum() < 10:
            continue
        a, b = ud[m], y[m]
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            continue
        corrs[i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    if np.all(np.isnan(corrs)):
        return min_lag, corrs
    best = int(np.nanargmax(np.abs(corrs)))
    return min_lag + best, corrs
