"""Objective -- model_spec.md §34, eq (65).

    (65)  min J = -c1 Throughput(u) + c2 Energy(u) + c3 S_sev(u) + c4 ||du||^2
          s.t. (u, w) in C

Quality and every specification live in ``C``, **never** in J (§34 / ТЗ
"Ключевой принцип"): an inadmissible regime must not be compensated by
economics.  ``assert_quality_not_in_objective`` is called by the solver and
raises if a caller ever tries to price a spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: names that identify a quality/spec quantity; they may never enter J
QUALITY_TERMS = {"sulfur", "s_prod", "mg.sulfur", "t95", "cfpp", "cloudpoint",
                 "flashpoint", "cetanenumber", "d15", "density15", "quality",
                 "spec", "margin"}


class QualityInObjectiveError(ValueError):
    """Raised when a quality quantity is priced into J (§34)."""


def assert_quality_not_in_objective(term_names) -> None:
    bad = [n for n in term_names
           if any(q in str(n).lower() for q in QUALITY_TERMS)]
    if bad:
        raise QualityInObjectiveError(
            f"quality/spec terms {bad} may not appear in the objective; they "
            "belong to the constraint set C (model_spec.md §34)")


@dataclass
class ObjectiveWeights:
    c1_throughput: float = 1.0
    c2_energy: float = 0.3
    c3_severity: float = 2.0
    c4_move: float = 0.1

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class EconomicModel:
    """Linear proxies for throughput and energy (ТЗ: a transparent cost proxy).

    ``throughput_coef`` and ``energy_coef`` map the manipulated vector to a
    scalar; both are [DOP] proxies documented in the assumptions register,
    because the package carries no economic data.
    """
    names: list[str]
    throughput_coef: np.ndarray
    energy_coef: np.ndarray
    throughput_unit: str = "t/h (proxy)"
    energy_unit: str = "MW-equivalent (proxy)"
    status: str = "[DOP]"
    note: str = ""

    def __post_init__(self):
        assert_quality_not_in_objective(self.names)
        self.throughput_coef = np.asarray(self.throughput_coef, float).ravel()
        self.energy_coef = np.asarray(self.energy_coef, float).ravel()
        if (self.throughput_coef.size != len(self.names)
                or self.energy_coef.size != len(self.names)):
            raise ValueError("coefficient length must match the channel list")

    def throughput(self, u) -> float:
        return float(np.asarray(u, float).ravel() @ self.throughput_coef)

    def energy(self, u) -> float:
        return float(np.asarray(u, float).ravel() @ self.energy_coef)


def objective_value(u, u_now, econ: EconomicModel, severity: float,
                    weights: ObjectiveWeights, du_scale=None) -> dict:
    """J of eq (65) with its breakdown, for the operator report."""
    u = np.asarray(u, float).ravel()
    u0 = np.asarray(u_now, float).ravel()
    sc = np.ones_like(u) if du_scale is None else np.asarray(du_scale, float)
    sc = np.where(sc > 0, sc, 1.0)
    du = (u - u0) / sc
    terms = {
        "throughput": -weights.c1_throughput * econ.throughput(u),
        "energy": weights.c2_energy * econ.energy(u),
        "severity": weights.c3_severity * float(severity),
        "move": weights.c4_move * float(du @ du),
    }
    return {"J": float(sum(terms.values())), "terms": terms,
            "throughput_proxy": econ.throughput(u),
            "energy_proxy": econ.energy(u)}


def pareto_front(candidates: list[dict], keys: tuple[str, ...] =
                 ("throughput_proxy", "energy_proxy", "S_sev"),
                 senses: tuple[int, ...] = (+1, -1, -1)) -> list[int]:
    """Indices of the non-dominated candidates (§34 alternative to weights).

    ``senses[i] = +1`` means larger is better for ``keys[i]``.
    """
    n = len(candidates)
    pts = np.array([[float(c.get(k, np.nan)) * s
                     for k, s in zip(keys, senses)] for c in candidates])
    keep = []
    for i in range(n):
        if not np.isfinite(pts[i]).all():
            continue
        dominated = False
        for j in range(n):
            if i == j or not np.isfinite(pts[j]).all():
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep
