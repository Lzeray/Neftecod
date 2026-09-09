"""Typed interfaces between the agents -- ТЗ §3; model_spec.md §36.

The agents communicate **only** through these dataclasses.  That is what makes
the multi-agent structure real rather than decorative: the orchestrator can be
run with any agent replaced by a stub that returns one of these objects, which
is exactly what tests/test_agents.py does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlantState:
    """Everything the agents are allowed to read about "now"."""
    timestamp: str
    u_now: dict[str, float]
    d_now: dict[str, float]
    derived: dict[str, float]
    freshness: dict[str, float]          # source -> age in hours
    flags: dict[str, bool]               # stale / outlier / missing
    raw_row: dict = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Returned by the quality agent (ТЗ: forecast, risk, confidence)."""
    timestamp: str
    y_hat: dict[str, float]              # predicted quality per parameter
    sigma_pred: dict[str, float]         # eq 58
    eps_plus: dict[str, float]           # eq 60 set-based bound
    p_violation: dict[str, float]        # Pr(spec violated)
    backoff: dict[str, float]            # z_{1-alpha} sigma_pred
    sensitivities: dict[str, float]      # d y_S / d u  per manipulated channel
    grad_d: dict[str, float]             # d y_S / d d  for the robust term
    confidence: str                      # "high" | "medium" | "low"
    horizon: dict = field(default_factory=dict)   # eq 32-33 response per channel
    data_quality: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ReliabilityAssessment:
    """Returned by the reliability agent (ТЗ: risk index, factors, limits)."""
    timestamp: str
    severity: float
    severity_class: str                  # "normal" | "elevated" | "severe"
    contributions: dict[str, float]
    deactivation: dict
    bound_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    """One admissible change of regime."""
    name: str
    u: dict[str, float]
    du: dict[str, float]
    y_hat: dict[str, float]
    metrics: dict[str, float]
    feasible: bool
    checks: list = field(default_factory=list)
    explanation: str = ""


@dataclass
class OptimizationOutcome:
    """Returned by the optimisation agent (ТЗ: scenarios, metrics, action)."""
    timestamp: str
    status: str                          # "optimal" | "infeasible" | "error"
    scenarios: list[Scenario] = field(default_factory=list)
    recommended: Scenario | None = None
    pareto_indices: list[int] = field(default_factory=list)
    reason: str = ""
    solver: str = ""
    out_of_range: list = field(default_factory=list)


@dataclass
class Decision:
    """Final operator-facing result (ТЗ §5 report blocks)."""
    timestamp: str
    action: str                          # "recommend" | "hold" | "refuse"
    headline: str
    state_block: dict
    risk_block: dict
    action_block: dict
    effect_block: dict
    constraint_block: dict
    confidence_block: dict
    explanation: str
    alternatives: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def conv(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        import dataclasses
        return {k: conv(v) for k, v in dataclasses.asdict(self).items()}
