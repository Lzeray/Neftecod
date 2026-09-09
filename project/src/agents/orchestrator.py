"""Orchestrator -- ТЗ §3; model_spec.md §36, eq (66).

    (66)  a recommendation is issued only if
              exists (u, w) in C            (the admissible set is not empty)
          AND (tau_LIMS <= tau_max OR ПАК valid)   (data freshness, eq 53-54)
          AND sigma_pred <= sigma_max               (confidence, eq 58)
          otherwise: a justified refusal.

The orchestrator asks each agent through its declared interface only, resolves
the conflict between them (quality and equipment risk outrank economics --
ТЗ "Ключевой принцип"), and persists the whole decision trace so the reasoning
can be audited afterwards.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.agents.contracts import (Decision, OptimizationOutcome, PlantState,
                                  QualityAssessment, ReliabilityAssessment)


class Orchestrator:
    name = "orchestrator"

    def __init__(self, quality_agent, reliability_agent, optimization_agent,
                 cfg, artifacts_dir: Path | None = None,
                 pak_usable: bool = True, pak_reason: str = ""):
        self.qa = quality_agent
        self.ra = reliability_agent
        self.oa = optimization_agent
        self.cfg = cfg
        self.artifacts = Path(artifacts_dir) if artifacts_dir else None
        #: Whether the stream analyser may stand in for a stale laboratory
        #: result in eq 66.  Identification decides this, not configuration: a
        #: source the filter tuning rejected cannot rescue the freshness clause
        #: just because it is producing numbers.
        self.pak_usable = bool(pak_usable)
        self.pak_reason = pak_reason

    # ------------------------------------------------------------------
    def gate(self, qual: QualityAssessment, state: PlantState,
             opt: OptimizationOutcome | None) -> tuple[bool, list[str]]:
        """Evaluate eq (66); returns (allowed, list of failed conditions)."""
        fails = []
        tau = float(state.freshness.get("LIMS:sulfur", np.inf))
        tau_max = float(self.cfg["spec"]["tau_lims_max_h"])
        pak_fresh = (not state.flags.get("PAK:sulfur__stale", True)
                     and float(state.freshness.get("PAK:sulfur", np.inf))
                     <= float(self.cfg["spec"]["tau_pak_max_h"]))
        pak_ok = pak_fresh and self.pak_usable
        if not (tau <= tau_max or pak_ok):
            why = ""
            if pak_fresh and not self.pak_usable:
                why = (", and the stream analyser cannot stand in: "
                       + (self.pak_reason or "identification rejected it as a "
                                             "quality source"))
            elif not pak_fresh:
                why = ", and the stream analyser is stale or missing"
            fails.append(
                f"data freshness (eq 53-54): the last laboratory sulfur result "
                f"is {tau:.1f} h old, above the {tau_max:.0f} h limit" + why)
        sigma = float(qual.sigma_pred.get("sulfur", np.inf))
        s_max = float(self.cfg["spec"]["sigma_max"])
        if not sigma <= s_max:
            fails.append(f"confidence (eq 58): sigma_pred = {sigma:.2f} mg/kg "
                         f"exceeds the {s_max:.2f} mg/kg threshold")
        ex = (qual.data_quality or {}).get("extrapolation") or {}
        if ex.get("outside"):
            fails.append(
                "extrapolation: the operating point is outside the domain the "
                f"kernel was identified on (Mahalanobis d^2 = "
                f"{ex['mahalanobis2']:.1f} vs threshold {ex['threshold']:.0f}); "
                "the kinetic sensitivity used to build the advice is not "
                "supported by data here")
        if opt is not None and opt.status not in ("optimal", "mitigate"):
            fails.append("admissible set (§31): " +
                         (opt.reason or "the constraint set C is empty at the "
                                        "required confidence"))
        return (not fails), fails

    # ------------------------------------------------------------------
    def decide(self, state: PlantState, row, utilde_row, eta: float,
               P_eta: float, persist: bool = True) -> Decision:
        qual = self.qa.assess(state, row, utilde_row, eta, P_eta)
        rel = self.ra.assess(state, bounds=self.oa.bounds)
        opt = self.oa.propose(state, qual, rel)
        allowed, fails = self.gate(qual, state, opt)

        S_max = float(self.cfg["spec"]["S_max_mgkg"])
        y = float(qual.y_hat["sulfur"])
        bo = float(qual.backoff["sulfur"])

        state_block = {
            "timestamp": state.timestamp,
            "key_parameters": {k: round(float(v), 4)
                               for k, v in state.u_now.items()
                               if np.isfinite(v)},
            "derived": {k: round(float(v), 4) for k, v in state.derived.items()
                        if np.isfinite(v)},
            "freshness_h": {k: (None if not np.isfinite(v) else round(float(v), 2))
                            for k, v in state.freshness.items()},
            "flags": {k: bool(v) for k, v in state.flags.items()
                      if isinstance(v, (bool, np.bool_))},
        }
        risk_block = {
            "sulfur_forecast_mgkg": round(y, 3),
            "chance_constrained_value_mgkg": round(y + bo, 3),
            "limit_mgkg": S_max,
            "p_violation": round(float(qual.p_violation["sulfur"]), 4),
            "set_based_bound_mgkg": (None
                                     if not np.isfinite(qual.eps_plus.get("sulfur", np.nan))
                                     else round(float(qual.eps_plus["sulfur"]), 3)),
            "severity": round(rel.severity, 4),
            "severity_class": rel.severity_class,
            "severity_factors": {k: round(v, 4)
                                 for k, v in rel.contributions.items()},
            "deactivation_hypothesis": rel.deactivation,
        }
        conf_block = {
            "confidence": qual.confidence,
            "sigma_pred_mgkg": round(float(qual.sigma_pred["sulfur"]), 3),
            "backoff_mgkg": round(bo, 3),
            "warnings": qual.notes + rel.notes,
            "data_quality": {k: (None if isinstance(v, float) and not np.isfinite(v)
                                 else v)
                             for k, v in qual.data_quality.items()},
        }

        if not allowed:
            dec = Decision(
                timestamp=state.timestamp, action="refuse",
                headline="No reliable recommendation",
                state_block=state_block, risk_block=risk_block,
                action_block={"changes": [],
                              "reason_no_action": "; ".join(fails)},
                effect_block={},
                constraint_block={"gate_eq66": fails, "passed": False},
                confidence_block=conf_block,
                explanation=self._refusal_text(fails, qual, opt),
                alternatives=[self._sc(s) for s in opt.scenarios[:6]],
                trace=self._trace(qual, rel, opt))
            return self._persist(dec) if persist else dec

        rec = opt.recommended
        hold = next((s for s in opt.scenarios if s.name.startswith("hold")), None)
        act, changes = self._changes(rec, hold)
        if opt.status == "mitigate":
            # The hard limit cannot be certified at the required confidence.
            # The system must say so in the headline, not bury it (ТЗ §5).
            act = "mitigate"
        eff = {
            "sulfur_now_mgkg": round(y, 3),
            "sulfur_after_mgkg": round(float(rec.y_hat["sulfur"]), 3),
            "sulfur_after_with_backoff_mgkg": round(
                float(rec.y_hat["sulfur_with_backoff"]), 3),
            "margin_to_limit_mgkg": round(float(rec.metrics["sulfur_margin"]), 3),
            "throughput_proxy": round(float(rec.metrics["throughput_proxy"]), 3),
            "energy_proxy": round(float(rec.metrics["energy_proxy"]), 3),
            "severity_after": round(float(rec.metrics["S_sev"]), 4),
            "objective_J": round(float(rec.metrics["J"]), 4),
        }
        if rec.metrics.get("blend") is not None:
            eff["blend"] = rec.metrics["blend"]
        # When the effect actually arrives (eq 32-33).  A recommendation that
        # quotes only the eventual value is misleading on a unit whose quality
        # responds hours after the move.
        timing = {}
        for c in changes:
            h = (qual.horizon or {}).get(c["tag"])
            if not h:
                continue
            timing[c["tag"]] = {
                "transport_delay_h": round(float(h["delay_h"]), 2),
                "time_constant_h": round(float(h["tau_h"]), 2),
                "time_to_63pct_of_effect_h": (
                    None if not np.isfinite(h["time_to_63pct_h"] or np.nan)
                    else round(float(h["time_to_63pct_h"]), 2)),
                "effect_within_horizon_mgkg": round(
                    float(h["dS_per_unit_at_horizon"]) * float(c["delta"]), 4),
            }
        if timing:
            eff["response_timing"] = timing
            eff["horizon_h"] = round(
                float(self.cfg["horizon"]["H"]) * float(self.cfg.Ts) / 3600.0, 2)
        if hold is not None:
            eff["vs_hold"] = {
                "dJ": round(float(rec.metrics["J"] - hold.metrics["J"]), 4),
                "d_throughput": round(float(rec.metrics["throughput_proxy"]
                                            - hold.metrics["throughput_proxy"]), 3),
                "d_energy": round(float(rec.metrics["energy_proxy"]
                                        - hold.metrics["energy_proxy"]), 3),
                "d_sulfur": round(float(rec.y_hat["sulfur"]
                                        - hold.y_hat["sulfur"]), 3),
            }
        dec = Decision(
            timestamp=state.timestamp, action=act,
            headline=self._headline(act, changes, rec, S_max),
            state_block=state_block, risk_block=risk_block,
            action_block={"changes": changes,
                          "certifies_spec": act != "mitigate",
                          "rate_limited": True,
                          "note": "every change respects |du| <= du_max and the "
                                  "declared [DOP] model range (§31)"},
            effect_block=eff,
            constraint_block={"gate_eq66": [], "passed": True,
                              "checks": rec.checks,
                              "out_of_declared_range":
                                  getattr(opt, "out_of_range", [])},
            confidence_block=conf_block,
            explanation=self._explanation(rec, hold, opt, qual, rel),
            alternatives=[self._sc(s) for i, s in enumerate(opt.scenarios)
                          if i in opt.pareto_indices or s is hold][:6],
            trace=self._trace(qual, rel, opt))
        return self._persist(dec) if persist else dec

    # ------------------------------------------------------------------
    @staticmethod
    def _sc(s) -> dict:
        return {"name": s.name, "feasible": s.feasible, "du": s.du,
                "y_hat": s.y_hat, "metrics": s.metrics,
                "explanation": s.explanation}

    def _changes(self, rec, hold) -> tuple[str, list[dict]]:
        from src import domain
        changes = []
        for k, d in rec.du.items():
            b = self.oa.bounds.get(k)
            span = (b.hi - b.lo) if b else 1.0
            if abs(d) <= 1e-9 or abs(d) < 1e-3 * max(span, 1e-9):
                continue
            ch = domain.channel(k)
            changes.append({
                "tag": k, "label": ch.label if ch else k,
                "unit": ch.unit if ch else (b.unit if b else ""),
                "current": round(float(rec.u[k] - d), 4),
                "recommended": round(float(rec.u[k]), 4),
                "delta": round(float(d), 4),
                "direction": "increase" if d > 0 else "decrease",
            })
        return ("recommend" if changes else "hold"), changes

    @staticmethod
    def _headline(act, changes, rec, S_max) -> str:
        if act == "mitigate":
            conf = rec.metrics.get("achievable_confidence")
            tail = (f"; attainable confidence {100 * conf:.0f}%"
                    if conf is not None else "")
            if not changes:
                return (f"Specification NOT certified and no admissible move "
                        f"available: forecast {rec.y_hat['sulfur']:.2f} mg/kg "
                        f"vs the {S_max:.0f} mg/kg limit{tail}")
            first = changes[0]
            return (f"Risk reduction only -- {first['direction']} "
                    f"{first['label']} {first['current']} -> "
                    f"{first['recommended']} {first['unit']}; the "
                    f"{S_max:.0f} mg/kg limit is NOT certified{tail}")
        if act == "hold":
            return (f"Hold the current regime -- forecast sulfur "
                    f"{rec.y_hat['sulfur']:.2f} mg/kg with back-off "
                    f"{rec.y_hat['sulfur_with_backoff']:.2f} mg/kg against the "
                    f"{S_max:.0f} mg/kg limit")
        first = changes[0]
        more = f" (+{len(changes) - 1} more)" if len(changes) > 1 else ""
        return (f"{first['direction'].capitalize()} {first['label']} "
                f"{first['current']} -> {first['recommended']} "
                f"{first['unit']}{more}")

    def _explanation(self, rec, hold, opt, qual, rel) -> str:
        parts = []
        y = float(qual.y_hat["sulfur"])
        bo = float(qual.backoff["sulfur"])
        S_max = float(self.cfg["spec"]["S_max_mgkg"])
        parts.append(
            f"Predicted product sulfur is {y:.2f} mg/kg; with the "
            f"{bo:.2f} mg/kg statistical back-off (eq 63 at "
            f"{100 * (1 - float(self.cfg['spec']['alpha'])):.0f}% confidence) "
            f"the constrained value is {y + bo:.2f} against the {S_max:.0f} "
            "mg/kg hard limit.")
        parts.append(
            f"The offset anchored on the laboratory result contributes "
            f"{float(qual.y_hat['sulfur_offset']):+.2f} mg/kg (eq 51); the "
            f"physical kernel gives {float(qual.y_hat['sulfur_kernel']):.2f} "
            f"and the identified residual {float(qual.y_hat['sulfur_residual']):+.2f}.")
        mitigating = opt.status == "mitigate"
        if mitigating:
            conf = rec.metrics.get("achievable_confidence")
            parts.append(
                "The admissible set C is EMPTY at the required confidence: no "
                "rate-limited move keeps the chance constraint of eq 63. This "
                "advice is lexicographic stage 1 of eq 65 -- the largest "
                "available reduction of the violation -- and it does not "
                "certify the specification"
                + (f" (attainable confidence {100 * conf:.0f}%)."
                   if conf is not None else "."))
        if hold is not None and rec is not hold:
            dJ = float(rec.metrics["J"] - hold.metrics["J"])
            tail = ("" if mitigating
                    else " while keeping every hard constraint satisfied")
            parts.append(
                f"Against holding the regime it changes the objective by "
                f"{dJ:+.4f} (throughput "
                f"{rec.metrics['throughput_proxy'] - hold.metrics['throughput_proxy']:+.3f}, "
                f"energy proxy {rec.metrics['energy_proxy'] - hold.metrics['energy_proxy']:+.3f}, "
                f"severity {rec.metrics['S_sev'] - hold.metrics['S_sev']:+.4f}) "
                f"and the forecast sulfur by "
                f"{rec.y_hat['sulfur'] - hold.y_hat['sulfur']:+.3f} mg/kg"
                f"{tail}.")
        n_feas = sum(1 for s in opt.scenarios if s.feasible)
        parts.append(f"{n_feas} of {len(opt.scenarios)} evaluated moves satisfy "
                     "every hard constraint; quality is never traded for "
                     "throughput (§34).")
        if rel.severity_class != "normal":
            parts.append(f"Equipment severity is {rel.severity_class} "
                         f"({rel.severity:.3f}); the optimiser was constrained "
                         "not to increase it beyond the configured maximum.")
        moved = {k for k, v in (rec.du or {}).items() if abs(v) > 1e-12}
        late = [(k, v["delay_h"]) for k, v in (qual.horizon or {}).items()
                if k in moved and v["delay_h"] > 0.5]
        if late:
            worst = max(late, key=lambda kv: kv[1])
            parts.append(
                f"The effect is not immediate: the identified transport delay "
                f"on {worst[0]} is {worst[1]:.1f} h, so the quality response "
                "begins only after that (eq 32-33), and the expected effect "
                "below is quoted for a move that is held.")
        if qual.notes:
            parts.append("Data caveats: " + "; ".join(qual.notes) + ".")
        return " ".join(parts)

    @staticmethod
    def _refusal_text(fails, qual, opt) -> str:
        head = ("No reliable recommendation can be issued: "
                + "; ".join(fails) + ".")
        tail = (" The model still reports its best estimate of "
                f"{float(qual.y_hat['sulfur']):.2f} mg/kg with "
                f"sigma_pred = {float(qual.sigma_pred['sulfur']):.2f} mg/kg, "
                "but the decision rule of eq 66 forbids acting on it.")
        return head + tail

    def _trace(self, qual, rel, opt) -> dict:
        return {
            "quality_agent": asdict(qual),
            "reliability_agent": asdict(rel),
            "optimization_agent": {
                "status": opt.status, "solver": opt.solver,
                "reason": opt.reason,
                "out_of_range": getattr(opt, "out_of_range", []),
                "scenarios": [self._sc(s) for s in opt.scenarios],
                "pareto_indices": opt.pareto_indices,
            },
        }

    def _persist(self, dec: Decision) -> Decision:
        if self.artifacts is None:
            return dec
        out = Path(self.artifacts) / "decisions"
        out.mkdir(parents=True, exist_ok=True)
        stamp = dec.timestamp.replace(":", "").replace(" ", "T")
        path = out / f"decision_{stamp}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dec.to_dict(), fh, ensure_ascii=False, indent=2,
                      default=_json_default)
        return dec


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
