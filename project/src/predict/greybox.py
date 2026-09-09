"""Grey-box predictor -- model_spec.md §26, eq (55).

    (55)  y_hat(t) = g_phi(u_tilde, d, m)   physical / ВАК kernel
                   + r_psi(u_tilde, d, m)   residual (IV, Huber, regularised)
                   + eta(t)                 offset anchored on LIMS

Constraints on the residual: monotonicity/sign restrictions from physics, e.g.
d y_S / d T_WABT <= 0.  They are enforced by projecting the fitted residual
coefficients onto the admissible sign cone, so an out-of-domain input can never
make the model claim that heating the reactor raises the product sulfur.

All inputs are the *lagged and shaped* u_tilde of eq 32-34, never instantaneous
set points (§26 closing note).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ident.huber import huber_fit
from src.ident.iv import iv_fit, robust_iv_fit
from src.ident.wls import WLSResult


@dataclass
class SignConstraint:
    """Admissible sign of one residual coefficient (physics, §26)."""
    name: str
    sign: int          # -1 (must be <= 0), +1 (must be >= 0), 0 (free)
    reason: str = ""


@dataclass
class GreyBoxModel:
    feature_names: list[str]
    theta: np.ndarray                       # residual coefficients r_psi
    intercept: float = 0.0
    kernel_name: str = ""
    sigma_resid: float = float("nan")
    sign_constraints: list[SignConstraint] = field(default_factory=list)
    fit_report: dict = field(default_factory=dict)

    # -----------------------------------------------------------------
    def residual(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {X.shape[1]}")
        return X @ self.theta + self.intercept

    def predict(self, kernel_value, X, eta: float = 0.0) -> np.ndarray:
        """eq (55): kernel + residual + offset."""
        g = np.asarray(kernel_value, float).ravel()
        r = self.residual(X)
        if g.size == 1:
            g = np.full(r.shape, float(g[0]))
        if g.size != r.size:
            raise ValueError("kernel and feature rows disagree")
        return g + r + float(eta)

    def sensitivity(self, feature: str) -> float:
        """d y_hat / d feature -- the residual slope, used by the optimiser."""
        i = self.feature_names.index(feature)
        return float(self.theta[i])


def project_signs(theta: np.ndarray, names: list[str],
                  constraints: list[SignConstraint]) -> tuple[np.ndarray, list[str]]:
    """Project the coefficients onto the admissible sign cone.

    Euclidean projection onto ``{theta : sign_i * theta_i >= 0}`` is simply
    clipping the offending entries to zero: it is the closest admissible point,
    and it degrades to "this channel has no credited effect" rather than to a
    physically impossible one.
    """
    theta = np.asarray(theta, float).copy()
    violated = []
    idx = {n: i for i, n in enumerate(names)}
    for c in constraints:
        if c.sign == 0 or c.name not in idx:
            continue
        i = idx[c.name]
        if c.sign * theta[i] < 0:
            violated.append(f"{c.name}: fitted {theta[i]:+.4g}, physics requires "
                            f"sign {c.sign:+d} ({c.reason}) -> clipped to 0")
            theta[i] = 0.0
    return theta, violated


def fit_greybox(X, y, kernel_value=None, feature_names: list[str] | None = None,
                instruments=None, sign_constraints: list[SignConstraint] | None = None,
                lam: float = 1.0, huber_delta: float = 1.345,
                weights=None, theta0=None) -> GreyBoxModel:
    """Fit r_psi on the kernel residual (eq 55).

    When ``instruments`` are supplied the estimate goes through IV (eq 43) to
    remove the closed-loop bias, then a Huber refit (eq 41) suppresses outliers
    while keeping the IV estimate as the regularisation anchor theta_0.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    if kernel_value is None:
        target = y
        kname = "none"
    else:
        g = np.asarray(kernel_value, float).ravel()
        if g.size == 1:
            g = np.full(y.shape, float(g[0]))
        target = y - g
        kname = "supplied"

    Xa = np.column_stack([X, np.ones(len(X))])
    names_a = list(names) + ["_intercept"]
    m = np.isfinite(Xa).all(axis=1) & np.isfinite(target)
    if m.sum() <= Xa.shape[1]:
        raise ValueError("not enough complete rows to fit the residual model")

    report: dict = {"n_rows": int(m.sum()), "n_features": Xa.shape[1]}
    w = None if weights is None else np.asarray(weights, float)[m]

    if instruments is not None:
        # Closed-loop data: the estimator itself must be IV (eq 43), not a
        # least-squares fit merely anchored on an IV point -- with a small
        # ridge the anchor would carry no weight and the bias would return.
        Z = np.atleast_2d(np.asarray(instruments, float))[m]
        Z = np.column_stack([Z, np.ones(len(Z))])
        ivres = robust_iv_fit(Xa[m], target[m], Z, delta=huber_delta,
                              ridge=max(lam, 1e-8), weights=w)
        theta_full = ivres.theta
        report["estimator"] = "robust IV (eq 41 + eq 43)"
        report["iv_relevance"] = ivres.relevance
    else:
        res: WLSResult = huber_fit(Xa[m], target[m], delta=huber_delta,
                                   weights=w, lam=lam, theta0=theta0)
        theta_full = res.theta
        report["estimator"] = "Huber WLS (eq 39-41)"
    theta_proj, violated = project_signs(theta_full, names_a,
                                         sign_constraints or [])
    report["sign_violations"] = violated
    if violated:
        # Clipping a slope leaves the intercept fitted against the *unclipped*
        # model, which biases every prediction by the removed contribution.
        # Re-anchor it robustly on the projected coefficients.
        theta_proj[-1] = float(np.median(target[m] - Xa[m, :-1] @ theta_proj[:-1]))
        report["intercept_reanchored"] = True
    resid = target[m] - Xa[m] @ theta_proj
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    report["sigma_resid"] = sigma
    report["theta"] = theta_proj.tolist()
    return GreyBoxModel(feature_names=names, theta=theta_proj[:-1],
                        intercept=float(theta_proj[-1]), kernel_name=kname,
                        sigma_resid=sigma,
                        sign_constraints=sign_constraints or [],
                        fit_report=report)
