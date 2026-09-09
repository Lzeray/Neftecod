"""Reliability agent -- ТЗ §3; model_spec.md §29-30, block G.

Returns the severity index (eq 61) with its per-proxy breakdown, the catalyst
deactivation *hypothesis* (eq 62), and the bound tightenings those imply for
the optimiser.  Nothing here claims a measured equipment condition: the package
carries no damage labels, so every number is a declared proxy.
"""
from __future__ import annotations

import numpy as np

from src.agents.contracts import PlantState, ReliabilityAssessment
from src.reliability import deactivation as DA
from src.reliability.severity import SeverityIndex, thermal_feedback_term


class ReliabilityAgent:
    name = "reliability"

    def __init__(self, index: SeverityIndex, deactivation_report: dict, cfg):
        self.index = index
        self.deact = deactivation_report or {}
        self.cfg = cfg

    def assess(self, state: PlantState,
               bounds: dict | None = None) -> ReliabilityAssessment:
        values = dict(state.u_now)
        values.update(state.d_now)
        values.update(state.derived)
        # measurement-only proxies (e.g. the R-202 differential pressure)
        values.update({k: v for k, v in state.raw_row.items()
                       if isinstance(v, float)})
        out = self.index.evaluate(values)
        sev = float(out["S_sev"])
        s_max = float(self.cfg["optimize"]["severity_max"])
        klass = ("normal" if sev <= 0.5 * s_max
                 else "elevated" if sev <= s_max else "severe")

        notes = []
        if out["missing"]:
            notes.append("severity proxies without data at this timestamp: "
                         + ", ".join(out["missing"]))
        if out["dominant"]:
            notes.append(f"the dominant severity contribution is "
                         f"{out['dominant']} ({out['contributions'][out['dominant']]:.3f})")

        overrides: dict[str, tuple[float, float]] = {}
        if self.deact.get("available") and bounds:
            rep = DA.DeactivationReport(
                slope_c_per_1000h=float(self.deact.get("slope_c_per_1000h", 0.0)),
                p_value=float(self.deact.get("p_value", 1.0)),
                n=float(self.deact.get("n", 0)),
                span_h=float(self.deact.get("span_h", 0.0)),
                significant=bool(self.deact.get("significant", False)))
            for key in ("HDS:T5",):
                if key in bounds:
                    b = bounds[key]
                    hi, why = DA.tighten_temperature_bound(b.hi, rep)
                    if hi < b.hi:
                        overrides[key] = (b.lo, hi)
                        notes.append(why)
        if klass == "severe":
            notes.append("severity above the configured maximum: the optimiser "
                         "must not increase it further (§31)")
        return ReliabilityAssessment(
            timestamp=state.timestamp, severity=sev, severity_class=klass,
            contributions={k: float(v) for k, v in out["contributions"].items()},
            deactivation=self.deact, bound_overrides=overrides, notes=notes)
