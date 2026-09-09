"""Source hierarchy and LIMS anchoring -- model_spec.md §24, eq (51)-(52).

    (51)  e = y_LIMS - (y_model(t_k) + eta(t_k^-))
          K = P^- / (P^- + R_lab)
          eta(t_k^+) = eta(t_k^-) + K e
          P^+ = (1 - K) P^-
    (52)  eta(t) = eta(t_k^+),   P(t) = P^+ + Q (t - t_k)

Priority LIMS -> ПАК -> ВАК (§24 / ТЗ).  ПАК enters through the same scalar
filter with R_PAK >> R_lab and is switched off when flagged stale.  When the
sources disagree, the laboratory result is the control fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class SourceRank(IntEnum):
    """Lower value = higher priority (model_spec.md §24)."""
    LIMS = 0
    PAK = 1
    VAK = 2
    NONE = 3


@dataclass
class BiasFilter:
    """Scalar offset filter of eq (51)-(52).

    ``eta``   -- current offset estimate, same units as the measurement
    ``P``     -- offset variance
    ``q_per_h`` -- random-walk growth rate of eq 52
    """
    eta: float = 0.0
    P: float = 1.0
    q_per_h: float = 0.02
    last_update_h: float | None = None
    log: list = field(default_factory=list)

    # ---------------------------------------------------------------
    def propagate(self, t_h: float) -> None:
        """eq (52): between updates eta is held and its variance grows."""
        if self.last_update_h is None:
            self.last_update_h = float(t_h)
            return
        dt = float(t_h) - self.last_update_h
        if dt < 0:
            raise ValueError("time must not run backwards")
        self.P += self.q_per_h * dt
        self.last_update_h = float(t_h)

    def update(self, t_h: float, y_meas: float, y_model: float, R: float,
               source: SourceRank = SourceRank.LIMS,
               accept: bool = True, reason: str = "") -> dict:
        """eq (51): one anchoring update."""
        self.propagate(t_h)
        info = {"t_h": float(t_h), "source": source.name, "accepted": False,
                "reason": reason, "e": np.nan, "K": np.nan,
                "eta": self.eta, "P": self.P}
        if not accept or not np.isfinite(y_meas) or not np.isfinite(y_model):
            info["reason"] = reason or "measurement unavailable or rejected"
            self.log.append(info)
            return info
        e = float(y_meas) - (float(y_model) + self.eta)
        K = self.P / (self.P + float(R))
        self.eta = self.eta + K * e
        self.P = (1.0 - K) * self.P
        info.update(accepted=True, e=e, K=K, eta=self.eta, P=self.P,
                    reason=reason or "accepted")
        self.log.append(info)
        return info

    def value(self, t_h: float) -> tuple[float, float]:
        """(eta, P) at time ``t_h`` without consuming a measurement."""
        if self.last_update_h is None:
            return self.eta, self.P
        dt = max(float(t_h) - self.last_update_h, 0.0)
        return self.eta, self.P + self.q_per_h * dt


@dataclass
class SourceReading:
    name: str
    rank: SourceRank
    value: float
    age_h: float
    variance: float
    stale: bool = False
    valid: bool = True

    @property
    def usable(self) -> bool:
        return (self.valid and not self.stale and np.isfinite(self.value)
                and np.isfinite(self.age_h))


def select_source(readings: list[SourceReading], max_age_h: dict[str, float]
                  ) -> tuple[SourceReading | None, list[dict]]:
    """Pick the highest-priority usable, fresh source (§24 hierarchy).

    Returns the winner (or None) and an audit trail explaining every rejection.
    """
    trail = []
    best: SourceReading | None = None
    for r in sorted(readings, key=lambda r: (int(r.rank), r.age_h)):
        limit = max_age_h.get(r.name, max_age_h.get(r.rank.name, np.inf))
        ok = r.usable and r.age_h <= limit
        why = ("selected" if ok and best is None else
               "not usable" if not r.usable else
               f"too old ({r.age_h:.1f} h > {limit:.1f} h)" if r.age_h > limit
               else "lower priority than the selected source")
        trail.append({"source": r.name, "rank": r.rank.name, "value": r.value,
                      "age_h": r.age_h, "stale": r.stale, "usable": r.usable,
                      "decision": why})
        if ok and best is None:
            best = r
    return best, trail


def fuse(readings: list[SourceReading], y_model: float, bias: BiasFilter,
         t_h: float, max_age_h: dict[str, float]) -> dict:
    """One fusion cycle: choose the source, anchor the offset, report.

    A LIMS reading always overrides a conflicting ПАК reading (§24).  The
    returned dict is fully auditable and is what the quality agent persists.
    """
    winner, trail = select_source(readings, max_age_h)
    if winner is None:
        eta, P = bias.value(t_h)
        return {"source": None, "eta": eta, "P_eta": P, "trail": trail,
                "y_fused": y_model + eta, "anchored": False,
                "reason": "no usable source -> model + held offset (eq 52)"}
    info = bias.update(t_h, winner.value, y_model, winner.variance,
                       source=winner.rank)
    eta, P = bias.value(t_h)
    conflict = None
    lims = [r for r in readings if r.rank == SourceRank.LIMS and r.usable]
    paks = [r for r in readings if r.rank == SourceRank.PAK and r.usable]
    if lims and paks:
        gap = abs(lims[0].value - paks[0].value)
        conflict = {"lims": lims[0].value, "pak": paks[0].value, "gap": gap,
                    "resolved_by": "LIMS (control fact, §24)"}
    return {"source": winner.name, "eta": eta, "P_eta": P, "trail": trail,
            "y_fused": y_model + eta, "anchored": info["accepted"],
            "innovation": info["e"], "gain": info["K"], "conflict": conflict,
            "reason": info["reason"]}
