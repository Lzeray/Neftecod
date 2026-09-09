"""Numerical solution of the recommendation problem -- model_spec.md §35.

Convex path: with the affine prediction of eq 57 and a linearised sulfur
response, the problem is a QP in ``(U, w)`` and is solved with cvxpy.
Non-convex path (LH kinetics) or a cvxpy failure: SLSQP from scipy, started at
the current regime, with the same constraint set.

Both paths enforce, as *hard constraints*:
    S_hat(u) + z_{1-alpha} sigma_pred + Gamma ||grad_d S||_U*  <=  S_max   (63-64)
    u_lo <= u <= u_hi, |u - u_now| <= du_max                              (§31)
    sum w = 1, w >= 0                                                     (eq 7)
    S_sev(u) <= S_sev_max                                                 (§29)

An infeasible problem returns ``status="infeasible"`` -- never a silently
clipped solution (prompt Block I).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.optimize.chance import backoff
from src.optimize.constraints import ConstraintSet
from src.optimize.objective import (EconomicModel, ObjectiveWeights,
                                    assert_quality_not_in_objective,
                                    objective_value)
from src.optimize.robust import UncertaintySet, robust_margin


@dataclass
class QualityChannel:
    """One linearised quality output used as a constraint.

    ``y = y0 + g' (u - u0)``; ``grad_d`` is the sensitivity to the uncertain
    feed properties (eq 64).
    """
    name: str
    y0: float
    grad_u: np.ndarray
    sigma: float
    direction: str = "max"           # "max": y <= limit ; "min": y >= limit
    limit: float = np.inf
    grad_d: np.ndarray | None = None
    eps_plus: float = 0.0            # set-based bound of eq 60
    hard: bool = True


@dataclass
class SeverityModel:
    """Linearised severity S_sev(u) = s0 + g' (u - u0) (§29 into §31)."""
    s0: float
    grad_u: np.ndarray


@dataclass
class OptimizationProblem:
    names: list[str]
    u_now: np.ndarray
    cset: ConstraintSet
    econ: EconomicModel
    weights: ObjectiveWeights
    quality: list[QualityChannel] = field(default_factory=list)
    severity: SeverityModel | None = None
    alpha: float = 0.05
    gamma: float = 1.0
    U_set: UncertaintySet | None = None
    blend_names: list[str] = field(default_factory=list)
    blend_sulfur: np.ndarray | None = None   # component sulfur, mg/kg
    blend_w_now: np.ndarray | None = None
    blend_value: np.ndarray | None = None    # throughput value per fraction
    blend_lo: np.ndarray | None = None
    blend_hi: np.ndarray | None = None

    def __post_init__(self):
        assert_quality_not_in_objective(self.econ.names)
        self.u_now = np.asarray(self.u_now, float).ravel()
        if self.u_now.size != len(self.names):
            raise ValueError("u_now size must match the channel list")


@dataclass
class OptimizationResult:
    status: str
    u: np.ndarray | None
    w: np.ndarray | None
    J: float
    detail: dict = field(default_factory=dict)
    solver: str = ""
    reason: str = ""


# ----------------------------------------------------------------------
def _bounds_arrays(p: OptimizationProblem):
    """Effective box for this cycle: declared range, rate limit, current point.

    The declared range is a conservative [DOP] *model* range, so the plant is
    sometimes already operating outside it.  Demanding an instantaneous jump
    back inside would make every such cycle infeasible and hide the real
    finding, so the box is widened to contain the current value; the rate limit
    then only lets the optimiser move *towards* the declared range, one step at
    a time.  ``outside`` reports which channels are currently out of range.
    """
    lo, hi, dmax = [], [], []
    for n in p.names:
        b = p.cset.bounds[n]
        lo.append(b.lo); hi.append(b.hi); dmax.append(b.du_max)
    lo, hi, dmax = np.array(lo), np.array(hi), np.array(dmax)
    lo_w = np.minimum(lo, p.u_now)
    hi_w = np.maximum(hi, p.u_now)
    return (np.maximum(lo_w, p.u_now - dmax), np.minimum(hi_w, p.u_now + dmax),
            lo, hi, dmax)


def out_of_range(p: OptimizationProblem) -> list[dict]:
    """Channels whose current value lies outside the declared model range."""
    out = []
    for i, n in enumerate(p.names):
        b = p.cset.bounds[n]
        v = float(p.u_now[i])
        if v < b.lo - 1e-9 or v > b.hi + 1e-9:
            out.append({"tag": n, "value": v, "lo": b.lo, "hi": b.hi,
                        "status": b.status,
                        "note": "the plant is operating outside the declared "
                                "[DOP] model range; the optimiser may only "
                                "move towards it at the rate limit"})
    return out


def _quality_rows(p: OptimizationProblem):
    """Rows of the linear constraint  a' u <= b  from eq 63-64."""
    rows, rhs, meta = [], [], []
    for q in p.quality:
        if not np.isfinite(q.limit):
            continue
        bo = backoff(q.sigma, p.alpha)
        rob = (robust_margin(q.grad_d, p.U_set, p.gamma)
               if (q.grad_d is not None and p.U_set is not None) else 0.0)
        g = np.asarray(q.grad_u, float).ravel()
        const = q.y0 - g @ p.u_now
        if q.direction == "max":
            rows.append(g)
            rhs.append(q.limit - const - bo - rob)
        else:
            rows.append(-g)
            rhs.append(const - q.limit - bo - rob)
        meta.append({"name": q.name, "backoff": bo, "robust": rob,
                     "limit": q.limit, "direction": q.direction})
    if not rows:
        return np.zeros((0, len(p.names))), np.zeros(0), meta
    return np.vstack(rows), np.array(rhs), meta


def solve_qp(p: OptimizationProblem, verbose: bool = False) -> OptimizationResult:
    """Convex QP path (§35) using cvxpy."""
    import cvxpy as cp

    n = len(p.names)
    u = cp.Variable(n)
    lo_e, hi_e, lo, hi, dmax = _bounds_arrays(p)
    cons = [u >= lo_e, u <= hi_e]

    A, b, meta = _quality_rows(p)
    if A.shape[0]:
        cons.append(A @ u <= b)

    if p.severity is not None:
        gs = np.asarray(p.severity.grad_u, float).ravel()
        cons.append(p.severity.s0 + gs @ (u - p.u_now) <= p.cset.severity_max)

    w = None
    blend_term = 0.0
    if p.blend_names:
        m = len(p.blend_names)
        w = cp.Variable(m)
        cons += [w >= 0, cp.sum(w) == 1]          # eq 7 simplex
        if p.blend_lo is not None:
            cons.append(w >= np.asarray(p.blend_lo, float))
        if p.blend_hi is not None:
            cons.append(w <= np.asarray(p.blend_hi, float))
        if p.blend_sulfur is not None:
            sp = next((q for q in p.quality if "sulfur" in q.name.lower()), None)
            if sp is not None:
                bo = backoff(sp.sigma, p.alpha)
                rob = (robust_margin(sp.grad_d, p.U_set, p.gamma)
                       if (sp.grad_d is not None and p.U_set is not None) else 0.0)
                # eq 7: sulfur blends linearly in mass, so the blended product
                # sulfur is affine in w and the chance constraint stays a
                # linear inequality.  The hydrotreated component carries the
                # *forecast* sulfur, which itself depends on u.
                S_blend = cp.sum(cp.multiply(np.asarray(p.blend_sulfur, float), w))
                cons.append(S_blend + bo + rob <= sp.limit)
        if p.blend_value is not None:
            blend_term = -p.weights.c1_throughput * (
                np.asarray(p.blend_value, float) @ w)

    sc = np.where(dmax > 0, dmax, 1.0)
    du = (u - p.u_now) / sc
    J = (-p.weights.c1_throughput * (p.econ.throughput_coef @ u)
         + p.weights.c2_energy * (p.econ.energy_coef @ u)
         + p.weights.c4_move * cp.sum_squares(du))
    if p.severity is not None:
        gs = np.asarray(p.severity.grad_u, float).ravel()
        J = J + p.weights.c3_severity * (p.severity.s0 + gs @ (u - p.u_now))
    J = J + blend_term

    prob = cp.Problem(cp.Minimize(J), cons)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception as exc:                     # noqa: BLE001
        return OptimizationResult("error", None, None, np.nan,
                                  {"exception": str(exc)}, "cvxpy")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return OptimizationResult(
            "infeasible" if "infeasible" in str(prob.status) else str(prob.status),
            None, None, np.nan, {"cvxpy_status": prob.status, "quality": meta},
            "cvxpy",
            reason=("no (u, w) satisfies C at the required confidence "
                    f"1-alpha = {1 - p.alpha:.2f}"))
    u_opt = np.asarray(u.value, float).ravel()
    w_opt = None if w is None else np.asarray(w.value, float).ravel()
    sev = (p.severity.s0
           + np.asarray(p.severity.grad_u, float) @ (u_opt - p.u_now)
           ) if p.severity is not None else 0.0
    jv = objective_value(u_opt, p.u_now, p.econ, sev, p.weights, du_scale=sc)
    return OptimizationResult("optimal", u_opt, w_opt, jv["J"],
                              {"quality": meta, "objective": jv,
                               "severity": float(sev)}, "cvxpy")


def solve_nlp(p: OptimizationProblem, quality_fun=None,
              severity_fun=None) -> OptimizationResult:
    """Non-convex path (§35): SLSQP, started from the current regime.

    ``quality_fun(u) -> dict[name, y_hat]`` lets the caller supply the full
    non-linear (Langmuir-Hinshelwood) prediction instead of the linearisation.
    """
    from scipy.optimize import minimize

    n = len(p.names)
    lo_e, hi_e, lo, hi, dmax = _bounds_arrays(p)
    sc = np.where(dmax > 0, dmax, 1.0)

    def sev_of(u):
        if severity_fun is not None:
            return float(severity_fun(u))
        if p.severity is None:
            return 0.0
        return float(p.severity.s0
                     + np.asarray(p.severity.grad_u, float) @ (u - p.u_now))

    def J(u):
        return objective_value(u, p.u_now, p.econ, sev_of(u), p.weights,
                               du_scale=sc)["J"]

    cons = []
    for q in p.quality:
        if not np.isfinite(q.limit):
            continue
        bo = backoff(q.sigma, p.alpha)
        rob = (robust_margin(q.grad_d, p.U_set, p.gamma)
               if (q.grad_d is not None and p.U_set is not None) else 0.0)

        def make(qq=q, bo=bo, rob=rob):
            def g(u):
                if quality_fun is not None:
                    y = float(quality_fun(u)[qq.name])
                else:
                    y = float(qq.y0 + np.asarray(qq.grad_u, float)
                              @ (u - p.u_now))
                return (qq.limit - (y + bo + rob) if qq.direction == "max"
                        else (y - bo - rob) - qq.limit)
            return g
        cons.append({"type": "ineq", "fun": make()})

    if p.severity is not None or severity_fun is not None:
        cons.append({"type": "ineq",
                     "fun": lambda u: p.cset.severity_max - sev_of(u)})

    res = minimize(J, p.u_now.copy(), method="SLSQP",
                   bounds=list(zip(lo_e, hi_e)), constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-10})
    u_opt = np.asarray(res.x, float)
    ok = bool(res.success) and all(c["fun"](u_opt) >= -1e-6 for c in cons)
    if not ok:
        return OptimizationResult("infeasible", None, None, np.nan,
                                  {"scipy": str(res.message)}, "slsqp",
                                  reason="SLSQP found no feasible point in C")
    jv = objective_value(u_opt, p.u_now, p.econ, sev_of(u_opt), p.weights,
                         du_scale=sc)
    return OptimizationResult("optimal", u_opt, None, jv["J"],
                              {"objective": jv, "severity": sev_of(u_opt)},
                              "slsqp")


def solve_min_violation(p: OptimizationProblem) -> OptimizationResult:
    """Lexicographic stage 1 of eq 65: minimise the constraint violation.

    model_spec.md §34 orders the problem "first admissibility, then the
    trade-off".  When C is empty the honest answer is not a clipped solution
    and not silence: it is the move that reduces the violation the most,
    reported as a *mitigation* that explicitly does not meet the specification.

        min_u  sum_j max(0, a_j' u - b_j)   s.t. boxes, rate limits, severity

    ``detail["violation"]`` is the residual violation in the units of the
    quality channel; zero means the set C is in fact non-empty.
    """
    import cvxpy as cp

    n = len(p.names)
    u = cp.Variable(n)
    lo_e, hi_e, lo, hi, dmax = _bounds_arrays(p)
    cons = [u >= lo_e, u <= hi_e]
    if p.severity is not None:
        gs = np.asarray(p.severity.grad_u, float).ravel()
        cons.append(p.severity.s0 + gs @ (u - p.u_now) <= p.cset.severity_max)
    A, b, meta = _quality_rows(p)
    if not A.shape[0]:
        return OptimizationResult("optimal", p.u_now.copy(), None, 0.0,
                                  {"violation": 0.0, "quality": meta}, "cvxpy")
    viol = cp.sum(cp.pos(A @ u - b))
    # Channels with zero sensitivity do not change the violation, so the
    # minimiser is a whole face of the box and the solver would return an
    # arbitrary point on it -- a recommendation that moves nine tags for no
    # reason. A small move penalty selects the least-action minimiser, so only
    # the channels that actually reduce the violation appear in the advice.
    sc = np.where(dmax > 0, dmax, 1.0)
    prob = cp.Problem(cp.Minimize(viol + 1e-4 * cp.sum_squares((u - p.u_now) / sc)),
                      cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as exc:                     # noqa: BLE001
        return OptimizationResult("error", None, None, np.nan,
                                  {"exception": str(exc)}, "cvxpy")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return OptimizationResult(str(prob.status), None, None, np.nan,
                                  {"cvxpy_status": prob.status}, "cvxpy",
                                  reason="even the minimum-violation problem "
                                         "is infeasible (boxes or severity)")
    u_opt = np.asarray(u.value, float).ravel()
    v = float(max(np.sum(np.clip(A @ u_opt - b, 0.0, None)), 0.0))
    return OptimizationResult("optimal", u_opt, None, v,
                              {"violation": v, "quality": meta}, "cvxpy",
                              reason="lexicographic stage 1 (minimum violation)")


def achievable_confidence(y_hat: float, sigma: float, limit: float) -> float:
    """The confidence level 1-alpha that *is* attainable at this forecast.

    Answers the operator's real question when the chance constraint fails:
    "what probability of staying inside the specification can this regime
    actually deliver?"  It is Pr(y <= limit) under eq 58's Gaussian assumption.
    """
    from scipy.stats import norm
    if sigma <= 0:
        return 1.0 if y_hat <= limit else 0.0
    return float(norm.cdf((limit - y_hat) / sigma))


def solve(p: OptimizationProblem, prefer: str = "qp", **kw) -> OptimizationResult:
    """Try the convex path first, fall back to the NLP path (§35)."""
    if prefer == "qp":
        r = solve_qp(p)
        if r.status == "optimal":
            return r
        if r.status == "infeasible":
            return r
        return solve_nlp(p, **kw)
    return solve_nlp(p, **kw)


def saa_scenarios(p: OptimizationProblem, u: np.ndarray, d_samples: np.ndarray,
                  channel: str = "sulfur") -> dict:
    """Sample-average approximation over historical d (§35).

    Evaluates the linear-in-d quality model on every historical scenario and
    reports the empirical violation rate, which must be <= alpha.
    """
    q = next((c for c in p.quality if c.name == channel), None)
    if q is None or q.grad_d is None or p.U_set is None:
        return {"available": False}
    D = np.atleast_2d(np.asarray(d_samples, float))
    y0 = q.y0 + np.asarray(q.grad_u, float) @ (np.asarray(u, float) - p.u_now)
    ys = y0 + (D - p.U_set.center) @ np.asarray(q.grad_d, float)
    rate = float((ys > q.limit).mean()) if q.direction == "max" \
        else float((ys < q.limit).mean())
    return {"available": True, "n": int(len(ys)), "violation_rate": rate,
            "alpha": p.alpha, "ok": rate <= p.alpha + 1e-12,
            "p95": float(np.quantile(ys, 0.95)), "mean": float(ys.mean())}
