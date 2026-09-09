"""Isolation checks for Block I -- model_spec.md §31-35 (eq 63-65).

Targets:
  * chance.check_max (eq 63)  -- analytic feasible region on a toy instance
  * robust (eq 64)            -- worst case verified by sampling U
  * objective (eq 65)         -- quality may never be priced into J
  * solve (§35)               -- larger sigma shrinks the feasible set and
    pushes S lower; an infeasible problem returns *infeasible*; quality cannot
    be traded for throughput
"""
import numpy as np
import pytest

from src.optimize import chance, constraints, objective, robust, solve
from src.optimize.constraints import Bound, ConstraintSet, SpecLimit
from src.optimize.objective import (EconomicModel, ObjectiveWeights,
                                    QualityInObjectiveError)
from src.optimize.robust import UncertaintySet
from src.optimize.solve import (OptimizationProblem, QualityChannel,
                                SeverityModel)


# ------------------------------------------------------- eq 63
def test_chance_constraint_analytic_region():
    # S_hat + 1.645 sigma <= 10  ->  with sigma = 1, S_hat <= 8.355
    c = chance.check_max(8.0, 1.0, 10.0, 0.05)
    assert c.z == pytest.approx(1.6448536, rel=1e-6)
    assert c.value == pytest.approx(9.6448536, rel=1e-6)
    assert c.satisfied
    assert not chance.check_max(8.4, 1.0, 10.0, 0.05).satisfied
    # the violation probability is exactly the normal tail
    assert c.p_violation == pytest.approx(0.02275, rel=1e-3)


def test_larger_sigma_forces_a_lower_S_hat():
    limits = []
    for sigma in (0.5, 1.0, 2.0, 4.0):
        s = 10.0 - chance.backoff(sigma, 0.05)
        limits.append(s)
    assert all(np.diff(limits) < 0), "more uncertainty must mean a lower target"


def test_chance_min_direction_and_set_based_check():
    c = chance.check_min(60.0, 2.0, 55.0, 0.05)
    assert c.satisfied and c.value == pytest.approx(60.0 - 1.6448536 * 2.0)
    assert not chance.check_min(56.0, 2.0, 55.0, 0.05).satisfied
    sb = chance.set_based_check(9.0, 0.5, 10.0)
    assert sb["satisfied"] and sb["margin"] == pytest.approx(0.5)
    assert not chance.set_based_check(9.8, 0.5, 10.0)["satisfied"]


def test_saa_violation_rate():
    y = np.r_[np.full(95, 8.0), np.full(5, 11.0)]
    assert chance.saa_violation_rate(y, 10.0) == pytest.approx(0.05)


# ------------------------------------------------------- eq 64
def test_box_support_is_the_worst_case_over_U():
    U = UncertaintySet(center=np.array([850.0, 1.2]), kind="box",
                       half_widths=np.array([20.0, 0.3]))
    g = np.array([0.01, 3.0])
    got = U.support(g)
    rng = np.random.default_rng(0)
    D = U.sample(20000, rng)
    sampled = float(np.max((D - U.center) @ g))
    assert sampled <= got + 1e-9
    assert sampled > 0.9 * got            # the bound is tight, not loose
    assert got == pytest.approx(20.0 * 0.01 + 0.3 * 3.0)


def test_ellipsoid_support_matches_the_l2_dual():
    P = np.array([[4.0, 1.0], [1.0, 9.0]])
    U = UncertaintySet(center=np.zeros(2), kind="ellipsoid", shape=P)
    g = np.array([1.0, -2.0])
    assert U.support(g) == pytest.approx(float(np.sqrt(g @ P @ g)))
    rng = np.random.default_rng(1)
    D = U.sample(20000, rng)
    assert float(np.max(D @ g)) <= U.support(g) + 1e-9


def test_robust_counterpart_is_feasible_for_every_sampled_d():
    U = UncertaintySet(center=np.array([850.0]), kind="box",
                       half_widths=np.array([25.0]))
    grad_d = np.array([0.02])            # mg/kg per kg/m3
    y_nom = 8.0
    worst = robust.worst_case_value(y_nom, grad_d, U, gamma=1.0)
    rng = np.random.default_rng(2)
    D = U.sample(50000, rng)
    ys = y_nom + (D - U.center) @ grad_d
    assert np.all(ys <= worst + 1e-9)
    assert worst == pytest.approx(8.0 + 0.5)


def test_uncertainty_set_guards():
    with pytest.raises(ValueError):
        UncertaintySet(center=[0.0], kind="box")
    with pytest.raises(ValueError):
        UncertaintySet(center=[0.0], kind="box", half_widths=[-1.0])
    with pytest.raises(ValueError):
        UncertaintySet(center=[0.0], kind="banana", half_widths=[1.0])
    U = UncertaintySet(center=[0.0], kind="box", half_widths=[1.0])
    assert U.contains([0.5]) and not U.contains([2.0])
    with pytest.raises(ValueError):
        robust.robust_margin([1.0], U, gamma=-1.0)


# ------------------------------------------------------- eq 65
def test_quality_can_never_enter_the_objective():
    with pytest.raises(QualityInObjectiveError):
        objective.assert_quality_not_in_objective(["T_wabt", "sulfur_margin"])
    with pytest.raises(QualityInObjectiveError):
        EconomicModel(["feed", "S_prod"], [1.0, 0.0], [0.0, 1.0])
    objective.assert_quality_not_in_objective(["feed", "T_wabt", "steam"])


def test_objective_breakdown_and_move_penalty():
    econ = EconomicModel(["feed", "T_wabt"], [1.0, 0.0], [0.0, 0.5])
    w = ObjectiveWeights(1.0, 0.3, 2.0, 0.1)
    j0 = objective.objective_value([100.0, 350.0], [100.0, 350.0], econ, 0.0, w)
    j1 = objective.objective_value([100.0, 352.0], [100.0, 350.0], econ, 0.0, w)
    assert j0["terms"]["move"] == 0.0 and j1["terms"]["move"] > 0
    assert j1["terms"]["energy"] > j0["terms"]["energy"]
    assert j0["terms"]["throughput"] == pytest.approx(-100.0)


def test_pareto_front_keeps_non_dominated():
    cand = [
        {"throughput_proxy": 10, "energy_proxy": 5, "S_sev": 0.1},   # good
        {"throughput_proxy": 9, "energy_proxy": 6, "S_sev": 0.2},    # dominated
        {"throughput_proxy": 8, "energy_proxy": 2, "S_sev": 0.05},   # good
    ]
    front = objective.pareto_front(cand)
    assert 0 in front and 2 in front and 1 not in front


# ------------------------------------------------------- §35 solver
def _problem(sigma=0.5, S0=8.0, severity_grad=None, hi_T=375.0):
    names = ["T_wabt", "feed"]
    cset = ConstraintSet(
        bounds={
            "T_wabt": Bound("T_wabt", 340.0, hi_T, 2.0, "degC", "[DOP]"),
            "feed": Bound("feed", 200.0, 300.0, 10.0, "t/h", "[DOP]"),
        },
        specs={"sulfur": SpecLimit("sulfur", 10.0, "max", "mg/kg")},
        severity_max=1.0,
    )
    econ = EconomicModel(names, throughput_coef=[0.0, 1.0],
                         energy_coef=[0.5, 0.1])
    q = QualityChannel(
        name="sulfur", y0=S0, sigma=sigma, direction="max", limit=10.0,
        # hotter -> less sulfur; more feed -> more sulfur
        grad_u=np.array([-0.4, 0.05]),
        grad_d=np.array([0.02]),
    )
    sev = None if severity_grad is None else SeverityModel(
        s0=0.2, grad_u=np.asarray(severity_grad, float))
    U = UncertaintySet(center=np.array([850.0]), kind="box",
                       half_widths=np.array([10.0]))
    return OptimizationProblem(names=names, u_now=np.array([360.0, 250.0]),
                               cset=cset, econ=econ,
                               weights=ObjectiveWeights(1.0, 0.3, 2.0, 0.05),
                               quality=[q], severity=sev, alpha=0.05,
                               gamma=1.0, U_set=U)


def test_solver_returns_a_feasible_optimum_and_respects_du_max():
    p = _problem()
    r = solve.solve_qp(p)
    assert r.status == "optimal"
    assert abs(r.u[0] - 360.0) <= 2.0 + 1e-6
    assert abs(r.u[1] - 250.0) <= 10.0 + 1e-6
    # and the chance constraint actually holds at the optimum
    y = p.quality[0].y0 + p.quality[0].grad_u @ (r.u - p.u_now)
    bo = chance.backoff(p.quality[0].sigma, p.alpha)
    rob = robust.robust_margin(p.quality[0].grad_d, p.U_set, p.gamma)
    assert y + bo + rob <= 10.0 + 1e-6


def test_more_uncertainty_shrinks_the_feasible_set_and_lowers_S():
    ys = []
    for sigma in (0.2, 0.8, 1.2):
        p = _problem(sigma=sigma, S0=9.0)
        r = solve.solve_qp(p)
        assert r.status == "optimal", sigma
        ys.append(p.quality[0].y0 + p.quality[0].grad_u @ (r.u - p.u_now))
    assert ys[0] > ys[1] > ys[2], "a larger back-off must force a lower S_hat"
    # pushed far enough, uncertainty alone empties C -- and that is reported,
    # not clipped away (eq 63 + eq 66)
    r_bad = solve.solve_qp(_problem(sigma=2.5, S0=9.0))
    assert r_bad.status == "infeasible" and r_bad.u is None


def test_infeasible_problem_is_reported_not_clipped():
    # sulfur far above the limit, and the temperature move rate cannot fix it
    p = _problem(sigma=3.0, S0=20.0)
    r = solve.solve_qp(p)
    assert r.status == "infeasible"
    assert r.u is None, "an infeasible problem must not return a clipped u"
    assert "confidence" in r.reason or "C" in r.reason


def test_quality_cannot_be_traded_for_throughput():
    """More feed pays (c1) and raises sulfur.  With the sulfur constraint
    binding, the solver must stop at the constraint, never buy throughput by
    letting S rise above the limit."""
    p = _problem(sigma=0.5, S0=9.6)
    r = solve.solve_qp(p)
    assert r.status == "optimal"
    y = p.quality[0].y0 + p.quality[0].grad_u @ (r.u - p.u_now)
    total = y + chance.backoff(0.5, 0.05) + robust.robust_margin(
        p.quality[0].grad_d, p.U_set, 1.0)
    assert total <= 10.0 + 1e-6
    # relaxing the limit would let the solver earn more -> proof it is binding
    p2 = _problem(sigma=0.5, S0=9.6)
    p2.quality[0].limit = 50.0
    r2 = solve.solve_qp(p2)
    assert r2.J < r.J - 1e-6


def test_severity_constraint_binds():
    p = _problem(sigma=0.3, S0=8.0, severity_grad=[0.5, 0.0])
    p.cset.severity_max = 0.25
    r = solve.solve_qp(p)
    assert r.status == "optimal"
    sev = 0.2 + 0.5 * (r.u[0] - p.u_now[0])
    assert sev <= 0.25 + 1e-6


def test_blend_fractions_sum_to_one():
    p = _problem()
    p.blend_names = ["hydrotreated", "straight_run"]
    p.blend_sulfur = np.array([5.0, 300.0])
    r = solve.solve_qp(p)
    assert r.status == "optimal"
    assert r.w is not None and r.w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(r.w >= -1e-8)
    # the blended sulfur with the back-off must respect the limit
    assert float(p.blend_sulfur @ r.w) + chance.backoff(0.5, 0.05) <= 10.0 + 1e-6


def test_nlp_path_agrees_with_the_qp_on_a_linear_instance():
    p = _problem()
    rq = solve.solve_qp(p)
    rn = solve.solve_nlp(p)
    assert rn.status == "optimal"
    assert np.allclose(rq.u, rn.u, atol=1e-3)


def test_nlp_path_handles_a_nonlinear_quality_function():
    p = _problem(sigma=0.2, S0=8.0)

    def quality_fun(u):
        # Arrhenius-like: sulfur falls exponentially with temperature
        return {"sulfur": 8.0 * np.exp(-0.05 * (u[0] - 360.0))
                + 0.05 * (u[1] - 250.0)}

    r = solve.solve_nlp(p, quality_fun=quality_fun)
    assert r.status == "optimal"
    assert quality_fun(r.u)["sulfur"] + chance.backoff(0.2, 0.05) <= 10.0 + 1e-6


def test_saa_reports_the_violation_rate():
    p = _problem(sigma=0.5, S0=8.0)
    rng = np.random.default_rng(3)
    D = p.U_set.sample(2000, rng)
    out = solve.saa_scenarios(p, p.u_now, D)
    assert out["available"] and 0.0 <= out["violation_rate"] <= 1.0


# ------------------------------------------------------- constraint set
def test_bound_and_spec_reporting():
    b = Bound("T", 340.0, 375.0, 2.0)
    assert b.feasible(360.0)[0]
    assert not b.feasible(380.0)[0]
    assert not b.feasible(365.0, u_now=360.0)[0]     # violates du_max
    with pytest.raises(ValueError):
        Bound("T", 380.0, 375.0, 2.0)
    with pytest.raises(ValueError):
        Bound("T", 340.0, 375.0, 0.0)
    s = SpecLimit("sulfur", 10.0, "max")
    assert s.satisfied(8.0, backoff=1.0) and not s.satisfied(9.5, backoff=1.0)


def test_constraint_set_check_is_a_full_audit_record():
    cs = ConstraintSet(bounds={"T": Bound("T", 340.0, 375.0, 2.0)},
                       specs={"sulfur": SpecLimit("sulfur", 10.0, "max")},
                       severity_max=1.0)
    rep = cs.check({"T": 380.0}, u_now={"T": 360.0},
                   y_hat={"sulfur": 9.9}, backoff={"sulfur": 0.5},
                   w=np.array([0.6, 0.4]), severity=1.5)
    assert not rep["feasible"]
    kinds = {c["kind"] for c in rep["violations"]}
    assert {"bound", "spec", "severity"} <= kinds
    assert all("detail" in c for c in rep["checks"])


def test_unidentifiable_channels_are_excluded_from_manipulated():
    cs = ConstraintSet(bounds={"A": Bound("A", 0, 1, 0.1),
                               "B": Bound("B", 0, 1, 0.1)})
    assert set(cs.manipulated) == {"A", "B"}
    cs.exclude_unidentifiable(["B"], "fails PE test (eq 44)")
    assert cs.manipulated == ["A"]
    assert "excluded" in cs.bounds["B"].provenance


def test_model_range_is_narrower_than_history_and_labelled():
    rng = np.random.default_rng(4)
    s = rng.normal(360.0, 5.0, 5000)
    lo, hi, note = constraints.model_range_from_history(s)
    assert lo > s.min() and hi < s.max()
    assert "[DOP]" in note and "NOT an industrial limit" in note
    with pytest.raises(ValueError):
        constraints.model_range_from_history(s[:10])
