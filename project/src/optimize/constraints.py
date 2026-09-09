"""The constraint set C -- model_spec.md §31.

  * hard quality (chance form): Pr(S_prod <= 10) >= 1 - alpha
  * other scenario specs (T95, CFPP, flash, cetane, density) in the same form
  * blending balance: sum_i w_i = 1, w_i >= 0
  * boxes: u_lo <= u <= u_hi -- a *confirmed* range or an explicitly flagged
    [DOP] conservative model range.  A historical min/max is never presented as
    an industrial limit (ТЗ "Правило границ").
  * rate limits: |du| <= du_max
  * severity: S_sev <= S_sev_max
  * channels that failed the identifiability test of §19 are excluded from the
    manipulated set
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bound:
    """One manipulated variable's admissible range."""
    name: str
    lo: float
    hi: float
    du_max: float
    unit: str = ""
    status: str = "[DOP]"          # [DOP] model range | [CONFIRMED]
    provenance: str = ""
    identifiable: bool = True

    def __post_init__(self):
        if not np.isfinite(self.lo) or not np.isfinite(self.hi):
            raise ValueError(f"{self.name}: bounds must be finite")
        if self.lo > self.hi:
            raise ValueError(f"{self.name}: lo > hi")
        if self.du_max <= 0:
            raise ValueError(f"{self.name}: du_max must be positive")

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.lo, self.hi))

    def feasible(self, value: float, u_now: float | None = None) -> tuple[bool, str]:
        if not np.isfinite(value):
            return False, f"{self.name}: not a number"
        if value < self.lo - 1e-9 or value > self.hi + 1e-9:
            return False, (f"{self.name}={value:.4g} outside "
                           f"[{self.lo:.4g}, {self.hi:.4g}] {self.status}")
        if u_now is not None and abs(value - u_now) > self.du_max + 1e-9:
            return False, (f"{self.name}: |du|={abs(value - u_now):.4g} exceeds "
                           f"du_max={self.du_max:.4g}")
        return True, "ok"


@dataclass
class SpecLimit:
    """One quality specification, always a *constraint*, never an objective."""
    name: str
    limit: float
    direction: str           # "max" -> y <= limit ; "min" -> y >= limit
    unit: str = ""
    hard: bool = True
    status: str = "[DOP]"

    def margin(self, y_hat: float, backoff: float = 0.0) -> float:
        """Signed slack; >= 0 means satisfied with the back-off applied."""
        if self.direction == "max":
            return self.limit - (y_hat + backoff)
        if self.direction == "min":
            return (y_hat - backoff) - self.limit
        raise ValueError(f"{self.name}: direction must be 'max' or 'min'")

    def satisfied(self, y_hat: float, backoff: float = 0.0) -> bool:
        return self.margin(y_hat, backoff) >= 0.0


@dataclass
class ConstraintSet:
    bounds: dict[str, Bound] = field(default_factory=dict)
    specs: dict[str, SpecLimit] = field(default_factory=dict)
    severity_max: float = 1.0
    blend_components: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def manipulated(self) -> list[str]:
        """Only identifiable channels may be manipulated (§19, §31)."""
        return [n for n, b in self.bounds.items() if b.identifiable]

    def exclude_unidentifiable(self, names: list[str], reason: str = "") -> list[str]:
        removed = []
        for n in names:
            if n in self.bounds:
                self.bounds[n].identifiable = False
                self.bounds[n].provenance += f" | excluded: {reason}"
                removed.append(n)
        return removed

    def check(self, u: dict[str, float], u_now: dict[str, float] | None = None,
              y_hat: dict[str, float] | None = None,
              backoff: dict[str, float] | None = None,
              w: np.ndarray | None = None,
              severity: float | None = None) -> dict:
        """Full feasibility report -- the audit record required by the ТЗ."""
        checks, ok = [], True
        for name, val in u.items():
            b = self.bounds.get(name)
            if b is None:
                checks.append({"kind": "bound", "name": name, "ok": False,
                               "detail": "no declared bound"})
                ok = False
                continue
            good, why = b.feasible(val, None if u_now is None else u_now.get(name))
            checks.append({"kind": "bound", "name": name, "ok": good,
                           "detail": why, "status": b.status,
                           "value": val, "lo": b.lo, "hi": b.hi})
            ok &= good
        if y_hat:
            bo = backoff or {}
            for name, val in y_hat.items():
                sp = self.specs.get(name)
                if sp is None:
                    continue
                m = sp.margin(val, bo.get(name, 0.0))
                good = m >= 0.0
                checks.append({"kind": "spec", "name": name, "ok": good,
                               "detail": (f"{name}={val:.4g} {sp.direction} "
                                          f"{sp.limit:.4g} {sp.unit}, back-off "
                                          f"{bo.get(name, 0.0):.4g}, margin "
                                          f"{m:+.4g}"),
                               "margin": m, "hard": sp.hard})
                ok &= good or not sp.hard
        if w is not None:
            w = np.asarray(w, float)
            s_ok = bool(np.isclose(w.sum(), 1.0, atol=1e-6) and np.all(w >= -1e-9))
            checks.append({"kind": "blend", "name": "sum_w", "ok": s_ok,
                           "detail": f"sum(w)={w.sum():.6f}, min(w)={w.min():.6f}"})
            ok &= s_ok
        if severity is not None:
            s_ok = bool(severity <= self.severity_max + 1e-12)
            checks.append({"kind": "severity", "name": "S_sev", "ok": s_ok,
                           "detail": f"S_sev={severity:.4g} <= "
                                     f"{self.severity_max:.4g}"})
            ok &= s_ok
        return {"feasible": ok, "checks": checks,
                "violations": [c for c in checks if not c["ok"]]}


def model_range_from_history(series, q_lo: float = 0.05, q_hi: float = 0.95,
                             shrink: float = 0.9) -> tuple[float, float, str]:
    """Conservative [DOP] model range from history (ТЗ "Правило границ").

    Takes an inner quantile band and shrinks it towards the median, so the
    result is *narrower* than anything ever observed and cannot be mistaken for
    an industrial limit.  The returned note records that explicitly.
    """
    import pandas as pd
    s = pd.Series(series).astype(float).dropna()
    if len(s) < 50:
        raise ValueError("need at least 50 samples for a model range")
    lo, hi = float(s.quantile(q_lo)), float(s.quantile(q_hi))
    med = float(s.median())
    lo = med + shrink * (lo - med)
    hi = med + shrink * (hi - med)
    note = (f"[DOP] conservative model range: {shrink:.0%} of the "
            f"{q_lo:.0%}-{q_hi:.0%} historical band around the median. "
            "NOT an industrial limit; no confirmed range was supplied.")
    return lo, hi, note
