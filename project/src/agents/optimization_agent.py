"""Optimisation agent -- ТЗ §3; model_spec.md §31-35, blocks H/I.

Builds the constraint set from the identified bounds plus whatever the
reliability agent tightened, poses the chance-constrained problem of eq 63-65,
and returns a *set* of admissible scenarios with their metrics -- not just one
number -- so the orchestrator can compare and explain.

Quality never enters the objective (§34): it enters only as a constraint, and
``objective.assert_quality_not_in_objective`` enforces that at construction.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src import domain
from src.agents.contracts import (OptimizationOutcome, PlantState,
                                  QualityAssessment, ReliabilityAssessment,
                                  Scenario)
from src.optimize import solve as SOLVE
from src.optimize.chance import backoff, check_max, set_based_check
from src.optimize.constraints import Bound, ConstraintSet, SpecLimit
from src.optimize.objective import (EconomicModel, ObjectiveWeights,
                                    objective_value, pareto_front)
from src.optimize.robust import UncertaintySet, robust_margin
from src.optimize.solve import (OptimizationProblem, QualityChannel,
                                SeverityModel)


class OptimizationAgent:
    name = "optimization"

    def __init__(self, bounds: dict, cfg, u_set: UncertaintySet | None = None,
                 severity_index=None, d_scenarios: np.ndarray | None = None):
        self.bounds = bounds
        self.cfg = cfg
        self.u_set = u_set
        self.severity_index = severity_index
        self.d_scenarios = d_scenarios

    # ------------------------------------------------------------------
    def _constraint_set(self, rel: ReliabilityAssessment) -> ConstraintSet:
        bounds = {}
        for k, b in self.bounds.items():
            lo, hi = b.lo, b.hi
            if k in rel.bound_overrides:
                lo, hi = rel.bound_overrides[k]
            bounds[k] = Bound(name=b.name, lo=lo, hi=hi, du_max=b.du_max,
                              unit=b.unit, status=b.status,
                              provenance=b.provenance,
                              identifiable=b.identifiable)
        specs = {"sulfur": SpecLimit("sulfur",
                                     float(self.cfg["spec"]["S_max_mgkg"]),
                                     "max", "mg/kg", hard=True,
                                     status="[CONFIRMED] ТЗ hard limit")}
        return ConstraintSet(bounds=bounds, specs=specs,
                             severity_max=float(self.cfg["optimize"]["severity_max"]))

    #: [DOP] relative weights of the energy proxy, per channel.
    ENERGY_WEIGHTS = {"AVT:T55": 1.0, "HDS:T5": 1.0, "AVT:F29": 0.5,
                      "HDS:F2": 0.2}

    def _economics(self, names: list[str]) -> EconomicModel:
        """[DOP] transparent linear proxies -- the package has no cost data.

        Both proxies are expressed **per unit of the declared model range** of
        each channel, so a temperature in degC and a gas rate in nm3/h can be
        summed at all: one unit of either proxy means "the full admissible
        span of that channel".  Mixing raw engineering units would let the
        ~1e5 nm3/h recycle-gas tag swamp every other term and make J
        meaningless.
        """
        span = np.array([max(self.bounds[k].hi - self.bounds[k].lo, 1e-9)
                         if k in self.bounds else 1.0 for k in names])
        thr = np.array([1.0 if k == "HDS:F26" else 0.0 for k in names]) / span
        en = np.array([self.ENERGY_WEIGHTS.get(k, 0.0) for k in names]) / span
        return EconomicModel(
            names=names, throughput_coef=thr, energy_coef=en,
            throughput_unit="model-range units of HDS feed rate (proxy)",
            energy_unit="weighted model-range units (proxy)",
            status="[DOP]",
            note="Throughput proxy = HDS feed rate; energy proxy = weighted "
                 "fired-heater and reactor temperatures, stripping steam and "
                 "recycle-gas rate. Both are normalised by each channel's "
                 "declared model range so the sum is dimensionless. No "
                 "economic data are supplied with the package (ТЗ §4).")

    # ------------------------------------------------------------------
    def propose(self, state: PlantState, qual: QualityAssessment,
                rel: ReliabilityAssessment) -> OptimizationOutcome:
        cset = self._constraint_set(rel)
        names = [k for k in cset.manipulated if k in state.u_now
                 and np.isfinite(state.u_now[k])]
        if not names:
            return OptimizationOutcome(state.timestamp, "infeasible",
                                       reason="no identifiable manipulated "
                                              "channel has a valid value now")
        u_now = np.array([state.u_now[k] for k in names])
        econ = self._economics(names)
        weights = ObjectiveWeights(**{
            k: float(v) for k, v in self.cfg["optimize"]["weights"].items()})

        grad_u = np.array([qual.sensitivities.get(k, 0.0) for k in names])
        grad_d = (np.array([qual.grad_d.get("feed_sulfur_mgkg", 0.0)])
                  if self.u_set is not None else None)
        q = QualityChannel(
            name="sulfur", y0=float(qual.y_hat["sulfur"]),
            grad_u=grad_u, sigma=float(qual.sigma_pred["sulfur"]),
            direction="max", limit=float(self.cfg["spec"]["S_max_mgkg"]),
            grad_d=grad_d, eps_plus=float(qual.eps_plus.get("sulfur", 0.0)))

        sev_model = None
        if self.severity_index is not None:
            g = self._severity_gradient(names, state)
            sev_model = SeverityModel(s0=rel.severity, grad_u=g)

        prob = OptimizationProblem(
            names=names, u_now=u_now, cset=cset, econ=econ, weights=weights,
            quality=[q], severity=sev_model,
            alpha=float(self.cfg["spec"]["alpha"]),
            gamma=float(self.cfg["optimize"]["gamma_robust"]),
            U_set=self.u_set)
        blend = self._blend_setup(state, qual)
        if blend is not None:
            prob.blend_names = blend["names"]
            prob.blend_sulfur = blend["sulfur"]
            prob.blend_value = blend["value"]
            prob.blend_lo = blend["lo"]
            prob.blend_hi = blend["hi"]

        scenarios = self._candidate_scenarios(prob, q, rel, state)
        outside = SOLVE.out_of_range(prob)
        res = SOLVE.solve_qp(prob)
        status, best, reason = res.status, None, res.reason
        if res.status == "optimal":
            best = self._make_scenario("optimizer", prob, res.u, q, rel, state)
            best.explanation = ("minimises J of eq 65 subject to the full "
                                "constraint set C")
            if res.w is not None and blend is not None:
                w = np.asarray(res.w, float)
                best.metrics["blend"] = {
                    "fractions": {n: round(float(v), 4)
                                  for n, v in zip(blend["names"], w)},
                    "blended_sulfur_mgkg": round(
                        float(blend["sulfur"] @ w), 3),
                    "sum_w": round(float(w.sum()), 6),
                    "note": "sulfur blends linearly in mass (eq 7); the "
                            "straight-run component carries the measured HDS "
                            "feed sulfur, so its admissible fraction is what "
                            "the 10 mg/kg limit allows.",
                }
            scenarios.insert(0, best)
        else:
            # Lexicographic stage 1 (§34): C is empty, so report the best
            # available risk reduction instead of a clipped or silent answer.
            mv = SOLVE.solve_min_violation(prob)
            if mv.status == "optimal" and mv.u is not None:
                best = self._make_scenario("minimum-violation", prob, mv.u, q,
                                           rel, state)
                y_best = float(best.y_hat["sulfur"])
                conf = SOLVE.achievable_confidence(
                    y_best, q.sigma, q.limit)
                best.explanation = (
                    "lexicographic stage 1 of eq 65: the constraint set C is "
                    "empty at the required confidence, so this is the move "
                    "that reduces the specification violation the most. It "
                    f"does NOT certify the {q.limit:.0f} mg/kg limit: the "
                    f"attainable confidence is {100 * conf:.1f}% against the "
                    f"required {100 * (1 - prob.alpha):.0f}%.")
                best.metrics["achievable_confidence"] = conf
                best.metrics["residual_violation_mgkg"] = float(
                    mv.detail.get("violation", np.nan))
                scenarios.insert(0, best)
                status = "mitigate"
                reason = (res.reason or "") + " | " + best.explanation
            else:
                status = "infeasible"
                reason = mv.reason or res.reason
        feas = [s for s in scenarios if s.feasible]
        idx = pareto_front([s.metrics for s in feas]) if feas else []
        outcome = OptimizationOutcome(
            timestamp=state.timestamp, status=status,
            scenarios=scenarios, recommended=best,
            pareto_indices=idx, reason=reason, solver=res.solver)
        outcome.out_of_range = outside
        return outcome

    # ------------------------------------------------------------------
    def _blend_setup(self, state: PlantState, qual: QualityAssessment):
        """Blend components with their *measured* sulfur (§4, eq 6-7).

        Only the sulfur of the two components is grounded in data: the
        hydrotreated stream carries the forecast of eq 55 and the straight-run
        stream carries the laboratory sulfur of the HDS feed point, which is
        the same material before the reactor.  Every other blend property is a
        [DOP] placeholder and is reported as such.
        """
        cfgb = self.cfg.get("blend") or {}
        if not cfgb.get("enabled"):
            return None
        comps = cfgb.get("components") or []
        if len(comps) < 2:
            return None
        S_hds = float(qual.y_hat["sulfur"])
        S_sr = float(state.derived.get("feed_sulfur_mgkg", np.nan))
        if not np.isfinite(S_sr):
            S_sr = float(state.raw_row.get("LIMSFEED:sulfur", np.nan))
        if not np.isfinite(S_sr):
            return None
        sulfur, value, lo, hi, names = [], [], [], [], []
        for c in comps:
            names.append(c["name"])
            sulfur.append(S_hds if c.get("source") == "model" else S_sr)
            value.append(float(c.get("value", 1.0)))
            lo.append(float(c.get("w_lo", 0.0)))
            hi.append(float(c.get("w_hi", 1.0)))
        return {"names": names, "sulfur": np.array(sulfur),
                "value": np.array(value), "lo": np.array(lo),
                "hi": np.array(hi)}

    def _severity_gradient(self, names, state) -> np.ndarray:
        g = np.zeros(len(names))
        values = dict(state.u_now)
        values.update(state.derived)
        values.update({k: v for k, v in state.raw_row.items()
                       if isinstance(v, float)})
        for i, k in enumerate(names):
            term = next((t for t in self.severity_index.terms if t.name == k),
                        None)
            if term is None or k not in values or not np.isfinite(values[k]):
                continue
            h = max(abs(values[k]) * 1e-3, 1e-6)
            up = dict(values); up[k] = values[k] + h
            dn = dict(values); dn[k] = values[k] - h
            g[i] = ((self.severity_index.evaluate(up)["S_sev"]
                     - self.severity_index.evaluate(dn)["S_sev"]) / (2 * h))
        return g

    def _candidate_scenarios(self, prob, q, rel, state) -> list[Scenario]:
        """A small, explainable menu of moves the operator can compare."""
        out = []
        u0 = prob.u_now
        out.append(self._make_scenario("hold (no action)", prob, u0.copy(), q,
                                       rel, state,
                                       "keep the current regime unchanged"))
        for i, name in enumerate(prob.names):
            b = prob.cset.bounds[name]
            for sign, tag in ((+1, "increase"), (-1, "decrease")):
                u = u0.copy()
                u[i] = float(np.clip(u0[i] + sign * b.du_max, b.lo, b.hi))
                if abs(u[i] - u0[i]) < 1e-12:
                    continue
                out.append(self._make_scenario(
                    f"{tag} {name} by one rate-limited step", prob, u, q, rel,
                    state, f"single-channel move on {name}"))
        return out

    def _make_scenario(self, name, prob, u, q, rel, state, why="") -> Scenario:
        u = np.asarray(u, float)
        y = float(q.y0 + q.grad_u @ (u - prob.u_now))
        bo = backoff(q.sigma, prob.alpha)
        rob = (robust_margin(q.grad_d, prob.U_set, prob.gamma)
               if (q.grad_d is not None and prob.U_set is not None) else 0.0)
        sev = rel.severity
        if prob.severity is not None:
            sev = float(prob.severity.s0
                        + np.asarray(prob.severity.grad_u) @ (u - prob.u_now))
        rep = prob.cset.check(
            {k: float(v) for k, v in zip(prob.names, u)},
            u_now={k: float(v) for k, v in zip(prob.names, prob.u_now)},
            y_hat={"sulfur": y}, backoff={"sulfur": bo + rob}, severity=sev)
        sc_scale = np.array([prob.cset.bounds[k].du_max for k in prob.names])
        jv = objective_value(u, prob.u_now, prob.econ, sev, prob.weights,
                             du_scale=sc_scale)
        setchk = set_based_check(y, q.eps_plus, q.limit) if np.isfinite(q.eps_plus) \
            else {"satisfied": None}
        chk = check_max(y, q.sigma, q.limit, prob.alpha)
        return Scenario(
            name=name,
            u={k: float(v) for k, v in zip(prob.names, u)},
            du={k: float(v - w) for k, v, w in zip(prob.names, u, prob.u_now)},
            y_hat={"sulfur": y, "sulfur_with_backoff": y + bo + rob},
            metrics={"J": jv["J"], "throughput_proxy": jv["throughput_proxy"],
                     "energy_proxy": jv["energy_proxy"], "S_sev": sev,
                     "p_violation": chk.p_violation,
                     "sulfur_margin": q.limit - (y + bo + rob),
                     "set_based_ok": setchk["satisfied"]},
            feasible=bool(rep["feasible"]), checks=rep["checks"],
            explanation=why)
