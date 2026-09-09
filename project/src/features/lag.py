"""Dead time and first-order shaping -- model_spec.md §12-13, eq (32)-(34).

    (32)  G_j(s) = K_j/(tau_j s + 1) * exp(-theta_j s)
    (33)  z_{j,k+1} = a_j z_{j,k} + K_j (1-a_j) du_{j,k-n_j},   a_j = exp(-Ts/tau_j)
    (34)  zeta^{(i)}_{k+1} = zeta^{(i-1)}_k,  zeta^{(1)}_{k+1} = u_k

The *effective input* ``u_tilde`` fed to every predictor is the delayed and
first-order-shaped input of eq 33, never the instantaneous set point: quality
responds to what the plant saw hours ago (ТЗ, "запаздывание отклика").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pure_delay(u, n: int, fill=np.nan):
    """u_{k-n} as an array (eq 34).  ``n`` in samples, ``n >= 0``."""
    if n < 0:
        raise ValueError("delay must be non-negative")
    u = np.asarray(u, dtype=float)
    if n == 0:
        return u.copy()
    out = np.full_like(u, fill, dtype=float)
    if n < len(u):
        out[n:] = u[:-n]
    return out


def delay_states(u, n: int) -> np.ndarray:
    """Delay line zeta^(1..n) of eq 34, shape (len(u), n).

    Column ``i-1`` holds ``u_{k-i}``; feeding column ``n-1`` into a linear model
    reproduces a pure delay of ``n`` samples exactly, without leaving the linear
    model class.
    """
    u = np.asarray(u, dtype=float)
    return np.column_stack([pure_delay(u, i) for i in range(1, n + 1)]) \
        if n > 0 else np.empty((len(u), 0))


def first_order_filter(u, tau_s: float, Ts: float, z0: float = 0.0,
                       gain: float = 1.0) -> np.ndarray:
    """Discrete first-order lag of eq 33 (no delay), exact ZOH form.

        z_{k+1} = a z_k + K (1-a) u_k,   a = exp(-Ts/tau)

    ``tau_s <= 0`` degenerates to the static gain ``K*u`` (no inertia).
    Returns ``z_k`` aligned with ``u_k`` (i.e. ``z[0] = z0``).
    """
    u = np.asarray(u, dtype=float)
    if Ts <= 0:
        raise ValueError("Ts must be positive")
    if tau_s <= 0:
        return gain * u.copy()
    a = float(np.exp(-Ts / tau_s))
    if u.size == 0:
        return u.copy()
    # y_k = a y_{k-1} + gain (1-a) u_{k-1}: the strictly causal ZOH numerator
    # scipy.signal.cont2discrete produces for K/(tau s + 1).  Written through
    # lfilter it is exact and ~1000x faster than the scalar recursion, which
    # matters because the delay/tau grid search evaluates it thousands of
    # times over 189k samples.
    from scipy.signal import lfilter
    uu = np.where(np.isfinite(u), u, 0.0)
    zi = np.array([a * float(z0)])
    z, _ = lfilter([0.0, gain * (1.0 - a)], [1.0, -a], uu, zi=zi)
    z[0] = float(z0)
    return z


def foptd(u, tau_s: float, theta_s: float, Ts: float, gain: float = 1.0,
          z0: float = 0.0) -> np.ndarray:
    """Full FOPTD channel of eq 32/33: delay by round(theta/Ts) then filter."""
    n = int(round(theta_s / Ts))
    ud = pure_delay(u, n, fill=0.0)
    return first_order_filter(ud, tau_s, Ts, z0=z0, gain=gain)


def foptd_step_response(t_s, K: float, tau_s: float, theta_s: float,
                        du: float = 1.0) -> np.ndarray:
    """Closed-form step response of eq 33 (continuous time), for verification."""
    t = np.asarray(t_s, dtype=float)
    out = np.zeros_like(t)
    m = t >= theta_s
    out[m] = K * du * (1.0 - np.exp(-(t[m] - theta_s) / tau_s))
    return out


def effective_inputs(df: pd.DataFrame, channels: dict[str, dict],
                     Ts: float) -> pd.DataFrame:
    """Build ``u_tilde`` for a set of channels.

    ``channels`` maps a column name to ``{"tau_s": ..., "theta_s": ...}``.
    Output columns are named ``<col>~lag``.
    """
    out = pd.DataFrame(index=df.index)
    for col, spec in channels.items():
        if col not in df.columns:
            continue
        u = df[col].to_numpy(dtype=float)
        u = pd.Series(u).ffill().to_numpy()
        out[f"{col}~lag"] = foptd(u, float(spec.get("tau_s", 0.0)),
                                  float(spec.get("theta_s", 0.0)), Ts)
    return out
