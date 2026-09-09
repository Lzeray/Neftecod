"""Regime severity index -- model_spec.md §29, eq (61).

    (61)  S_sev(t) = sum_i omega_i [ (p_i(t) - p_i^ref) / p_i^scale ]_+^{gamma_i}

Proxies p_i (§29): furnace outlet temperature vs its limit, packing/bed
differential pressure, column and furnace loading vs design, steam rate, levels,
and the positive thermal feedback of §9.

[DOP] The whole index is a *proxy*: the data package carries no equipment
damage labels.  Every reference and scale is declared in the assumptions
register, and the index is reported alongside the individual contributions so
an operator can see which proxy drove it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SeverityTerm:
    """One proxy of eq (61)."""
    name: str
    ref: float                 # p_i^ref -- value below which there is no penalty
    scale: float               # p_i^scale
    weight: float = 1.0        # omega_i
    gamma: float = 2.0         # gamma_i
    direction: int = +1        # +1 penalise above ref, -1 penalise below
    status: str = "[DOP]"
    note: str = ""

    def contribution(self, value) -> np.ndarray:
        v = np.asarray(value, dtype=float)
        if self.scale <= 0:
            raise ValueError(f"{self.name}: scale must be positive")
        excess = self.direction * (v - self.ref) / self.scale
        return self.weight * np.power(np.clip(excess, 0.0, None), self.gamma)


@dataclass
class SeverityIndex:
    terms: list[SeverityTerm] = field(default_factory=list)

    def evaluate(self, values: dict[str, float]) -> dict:
        """S_sev and the per-proxy breakdown (eq 61)."""
        contribs, missing = {}, []
        total = 0.0
        for t in self.terms:
            if t.name not in values or not np.isfinite(values[t.name]):
                missing.append(t.name)
                continue
            c = float(t.contribution(values[t.name]))
            contribs[t.name] = c
            total += c
        return {"S_sev": total, "contributions": contribs, "missing": missing,
                "dominant": max(contribs, key=contribs.get) if contribs else None}

    def evaluate_frame(self, df: pd.DataFrame) -> pd.Series:
        out = np.zeros(len(df))
        for t in self.terms:
            if t.name in df.columns:
                c = t.contribution(df[t.name].to_numpy(dtype=float))
                out = out + np.nan_to_num(c, nan=0.0)
        return pd.Series(out, index=df.index, name="S_sev")


# ---------------------------------------------------------------- robust scaling
def robust_reference(series, q_ref: float = 0.90) -> tuple[float, float]:
    """Median/MAD-based (ref, scale) for a proxy without a documented limit.

    Uses the median and 1.4826*MAD so that a handful of frozen or spiked
    samples cannot move the reference -- §29 needs a *robust* standardisation.
    ``q_ref`` selects the quantile treated as the onset of a severe regime.

    [DOP] A historical quantile is **not** an industrial limit; it only sets a
    conservative model scale (ТЗ "Правило границ").
    """
    s = pd.Series(series).astype(float).dropna()
    if len(s) < 10:
        raise ValueError("need at least 10 samples for a robust reference")
    med = float(s.median())
    mad = 1.4826 * float((s - med).abs().median())
    if mad <= 0:
        mad = float(s.std(ddof=0))
    if mad <= 0:
        mad = 1.0
    ref = float(s.quantile(q_ref))
    return ref, mad


def thermal_feedback_term(a_TT: float, scale: float = 1e-4,
                          weight: float = 1.0) -> float:
    """Positive thermal feedback contribution of §9 to eq 61.

    ``a_TT > 0`` means the linearised temperature mode is unstable (exothermic
    runaway); it contributes quadratically, and a stable mode contributes zero.
    """
    return float(weight * max(a_TT / scale, 0.0) ** 2)
