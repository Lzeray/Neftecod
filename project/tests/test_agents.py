"""Isolation checks for Block J -- ТЗ §3; model_spec.md §36 (eq 66).

Targets:
  * the orchestrator works with *every* agent replaced by a stub returning the
    declared dataclass -- proof that the agents communicate only through their
    interfaces
  * the refusal branch fires when the LIMS result is stale, when sigma_pred is
    above the threshold, and when the constraint set is empty
  * every emitted recommendation carries the report blocks the ТЗ requires
"""
import numpy as np
import pytest

from src.agents.contracts import (Decision, OptimizationOutcome, PlantState,
                                  QualityAssessment, ReliabilityAssessment,
                                  Scenario)
from src.agents.orchestrator import Orchestrator
from src.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _state(lims_age=2.0, pak_stale=False):
    return PlantState(
        timestamp="2025-01-01 00:00:00",
        u_now={"HDS:T5": 365.0, "HDS:F26": 250.0},
        d_now={"AVT:F7": 250.0},
        derived={"derived:T_WABT_C": 363.0},
        freshness={"LIMS:sulfur": lims_age, "PAK:sulfur": 0.1},
        flags={"PAK:sulfur__stale": pak_stale})


def _qual(sigma=1.0, y=8.0):
    return QualityAssessment(
        timestamp="2025-01-01 00:00:00",
        y_hat={"sulfur": y, "sulfur_kernel": y, "sulfur_residual": 0.0,
               "sulfur_offset": 0.0},
        sigma_pred={"sulfur": sigma}, eps_plus={"sulfur": 0.5},
        p_violation={"sulfur": 0.02}, backoff={"sulfur": 1.645 * sigma},
        sensitivities={"HDS:T5": -0.3, "HDS:F26": 0.05},
        grad_d={"feed_sulfur_mgkg": 1e-4}, confidence="high")


def _rel(sev=0.1):
    return ReliabilityAssessment(
        timestamp="2025-01-01 00:00:00", severity=sev,
        severity_class="normal", contributions={"HDS:T5": sev},
        deactivation={"available": False})


def _scenario(name, feasible=True, J=1.0, y=8.0):
    return Scenario(
        name=name, u={"HDS:T5": 365.0, "HDS:F26": 250.0},
        du={"HDS:T5": 0.0, "HDS:F26": 0.0},
        y_hat={"sulfur": y, "sulfur_with_backoff": y + 1.6},
        metrics={"J": J, "throughput_proxy": 1.0, "energy_proxy": 0.5,
                 "S_sev": 0.1, "p_violation": 0.02, "sulfur_margin": 0.4,
                 "set_based_ok": True},
        feasible=feasible, checks=[{"kind": "spec", "name": "sulfur",
                                    "ok": feasible, "detail": "stub"}])


class StubQuality:
    def __init__(self, q): self.q = q
    def assess(self, *a, **k): return self.q


class StubReliability:
    def __init__(self, r): self.r = r
    def assess(self, *a, **k): return self.r


class StubOptimization:
    def __init__(self, o, bounds=None):
        self.o, self.bounds = o, bounds or {}
    def propose(self, *a, **k): return self.o


def _orch(cfg, qual=None, rel=None, opt=None):
    qual = qual or _qual()
    rel = rel or _rel()
    if opt is None:
        rec = _scenario("optimizer")
        opt = OptimizationOutcome("2025-01-01 00:00:00", "optimal",
                                  [rec, _scenario("hold (no action)")],
                                  recommended=rec, pareto_indices=[0])
    return Orchestrator(StubQuality(qual), StubReliability(rel),
                        StubOptimization(opt), cfg, artifacts_dir=None)


# ------------------------------------------------------------------
def test_orchestrator_runs_entirely_on_stubs(cfg):
    """The orchestrator must not reach into any agent's internals."""
    d = _orch(cfg).decide(_state(), row=None, utilde_row=None, eta=0.0,
                          P_eta=0.1, persist=False)
    assert isinstance(d, Decision)
    assert d.action in ("recommend", "hold")
    assert d.constraint_block["passed"]


def test_report_carries_every_block_the_tz_requires(cfg):
    d = _orch(cfg).decide(_state(), None, None, 0.0, 0.1, persist=False)
    # ТЗ §5: time+state, risk, action, expected effect, constraint check,
    # confidence, explanation
    assert d.state_block["timestamp"]
    assert "freshness_h" in d.state_block
    assert "sulfur_forecast_mgkg" in d.risk_block
    assert "severity_class" in d.risk_block
    assert "changes" in d.action_block
    assert "sulfur_after_mgkg" in d.effect_block
    assert "gate_eq66" in d.constraint_block
    assert d.confidence_block["confidence"] in ("high", "medium", "low")
    assert len(d.explanation) > 40
    assert "quality_agent" in d.trace and "optimization_agent" in d.trace


def test_refusal_when_lims_is_stale(cfg):
    tau_max = float(cfg["spec"]["tau_lims_max_h"])
    st = _state(lims_age=tau_max + 50.0, pak_stale=True)
    d = _orch(cfg).decide(st, None, None, 0.0, 0.1, persist=False)
    assert d.action == "refuse"
    assert any("freshness" in f for f in d.constraint_block["gate_eq66"])
    assert "laboratory" in d.explanation


def test_fresh_pak_rescues_a_stale_lims(cfg):
    tau_max = float(cfg["spec"]["tau_lims_max_h"])
    st = _state(lims_age=tau_max + 50.0, pak_stale=False)
    d = _orch(cfg).decide(st, None, None, 0.0, 0.1, persist=False)
    assert d.action != "refuse", "eq 66 allows a valid ПАК to stand in"


def test_refusal_when_sigma_is_too_large(cfg):
    s_max = float(cfg["spec"]["sigma_max"])
    d = _orch(cfg, qual=_qual(sigma=s_max + 5.0)).decide(
        _state(), None, None, 0.0, 0.1, persist=False)
    assert d.action == "refuse"
    assert any("confidence" in f for f in d.constraint_block["gate_eq66"])


def test_refusal_when_the_constraint_set_is_empty(cfg):
    opt = OptimizationOutcome("2025-01-01 00:00:00", "infeasible",
                              [_scenario("hold (no action)", feasible=False)],
                              recommended=None,
                              reason="no (u, w) satisfies C")
    d = _orch(cfg, opt=opt).decide(_state(), None, None, 0.0, 0.1,
                                   persist=False)
    assert d.action == "refuse"
    assert any("admissible set" in f for f in d.constraint_block["gate_eq66"])
    assert d.action_block["changes"] == []


def test_mitigate_is_reported_as_not_certifying_the_spec(cfg):
    rec = _scenario("minimum-violation", feasible=False, y=9.5)
    rec.metrics["achievable_confidence"] = 0.72
    opt = OptimizationOutcome("2025-01-01 00:00:00", "mitigate",
                              [rec, _scenario("hold (no action)", False)],
                              recommended=rec,
                              reason="C empty; best risk reduction")
    d = _orch(cfg, opt=opt).decide(_state(), None, None, 0.0, 0.1,
                                   persist=False)
    assert d.action == "mitigate"
    assert d.action_block["certifies_spec"] is False
    assert "NOT certified" in d.headline
    assert "EMPTY" in d.explanation or "empty" in d.explanation


def test_hold_when_the_optimum_is_the_current_regime(cfg):
    rec = _scenario("optimizer")           # du == 0 everywhere
    opt = OptimizationOutcome("2025-01-01 00:00:00", "optimal",
                              [rec, _scenario("hold (no action)")],
                              recommended=rec)
    d = _orch(cfg, opt=opt).decide(_state(), None, None, 0.0, 0.1,
                                   persist=False)
    assert d.action == "hold" and d.action_block["changes"] == []
    assert "Hold" in d.headline


def test_decision_is_json_serialisable(cfg):
    import json
    from src.agents.orchestrator import _json_default
    d = _orch(cfg).decide(_state(), None, None, 0.0, 0.1, persist=False)
    s = json.dumps(d.to_dict(), default=_json_default, ensure_ascii=False)
    assert len(s) > 200


def test_decision_is_persisted_when_requested(cfg, tmp_path):
    o = _orch(cfg)
    o.artifacts = tmp_path
    d = o.decide(_state(), None, None, 0.0, 0.1, persist=True)
    files = list((tmp_path / "decisions").glob("decision_*.json"))
    assert len(files) == 1 and files[0].stat().st_size > 200
