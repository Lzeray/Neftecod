"""Blending balances -- model_spec.md §4, §21, eq (6)-(9), (46).

    (6)  m_sum = sum_i m_i,     S_mix = sum_i m_i S_i / m_sum
    (7)  S_mix = sum_i p_i S_i, sum_i p_i = 1                (linear in mass)
    (8)  dS_mix = sum_i p_i0 dS_i + sum_i (S_i0 - S_mix0)/m_sum0 dm_i
    (9)  M dS/dt = sum_i m_i (S_i - S),   tau_mix = M / m_sum
    (46) non-linear properties through blending indices

Sulfur is the *linking* constraint of the whole problem and is linear in mass.
Flash point, CFPP, cloud point and T95 are **not** (model_spec.md §4 closing
note) -- attempting to blend them linearly raises.
"""
from __future__ import annotations

import numpy as np

from src.blend.indices import (INDEXED_PROPERTIES, LINEAR_PROPERTIES,
                               blend_indexed, get_index)


def normalise_fractions(w, tol: float = 1e-9) -> np.ndarray:
    """Project onto the simplex boundary: clip at 0 then renormalise to 1."""
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        raise ValueError("empty blend")
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= tol:
        raise ValueError("blend fractions are all zero")
    return w / total


def check_simplex(w, atol: float = 1e-8) -> None:
    w = np.asarray(w, dtype=float)
    if np.any(w < -atol):
        raise ValueError("blend fractions must be non-negative")
    if not np.isclose(w.sum(), 1.0, atol=atol):
        raise ValueError(f"blend fractions must sum to 1, got {w.sum()}")


def blend_sulfur_mass(m_dot, S) -> float:
    """S_mix from mass flows, eq (6)."""
    m = np.asarray(m_dot, dtype=float)
    s = np.asarray(S, dtype=float)
    tot = m.sum()
    if tot <= 0:
        raise ValueError("total mass flow must be positive")
    return float(np.sum(m * s) / tot)


def blend_linear(w, y) -> float:
    """y_mix = sum_i w_i y_i, eq (7).  Requires a valid simplex."""
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    check_simplex(w)
    if w.shape != y.shape:
        raise ValueError("shape mismatch")
    return float(np.sum(w * y))


def blend_property(prop: str, w, y, dose_ppm: float = 0.0,
                   depressant: dict | None = None) -> float:
    """Blend one property, choosing the *correct* rule from its name.

    Linear-in-mass properties (sulfur, density, cetane) use eq 7; everything
    registered in ``INDEXED_PROPERTIES`` uses eq 46.  A property that is
    neither is rejected rather than silently blended linearly.
    """
    if prop in LINEAR_PROPERTIES:
        return blend_linear(w, y)
    if prop in INDEXED_PROPERTIES:
        val = blend_indexed(w, y, get_index(prop))
        if dose_ppm and depressant and prop in ("CFPP", "CloudPoint"):
            from src.blend.indices import depressant_shift
            val += float(depressant_shift(dose_ppm, depressant["dmax_c"],
                                          depressant["k_ppm"]))
        return val
    raise KeyError(f"no blending rule registered for {prop!r}")


def delta_sulfur(p0, S0, S_mix0: float, m_sum0: float, dS, dm) -> float:
    """Local deviation form, eq (8)."""
    p0 = np.asarray(p0, float); S0 = np.asarray(S0, float)
    dS = np.asarray(dS, float); dm = np.asarray(dm, float)
    if m_sum0 <= 0:
        raise ValueError("m_sum0 must be positive")
    return float(np.sum(p0 * dS) + np.sum((S0 - S_mix0) / m_sum0 * dm))


def mixer_step(S, m_dot, S_in, M_kg: float, dt_s: float) -> float:
    """One explicit Euler step of the well-mixed tank, eq (9)."""
    m = np.asarray(m_dot, float)
    s_in = np.asarray(S_in, float)
    if M_kg <= 0:
        raise ValueError("mixer hold-up must be positive")
    return float(S + dt_s * np.sum(m * (s_in - S)) / M_kg)


def tau_mix(M_kg: float, m_sum_kg_s: float) -> float:
    """tau_mix = M / m_sum, eq (9)."""
    if m_sum_kg_s <= 0:
        raise ValueError("total mass flow must be positive")
    return float(M_kg / m_sum_kg_s)


def simulate_mixer(S0: float, m_dot, S_in, M_kg: float, dt_s: float,
                   n_steps: int) -> np.ndarray:
    """Trajectory of eq (9); converges to the eq 6 steady-state blend."""
    out = np.empty(n_steps + 1)
    out[0] = S0
    for k in range(n_steps):
        out[k + 1] = mixer_step(out[k], m_dot, S_in, M_kg, dt_s)
    return out
