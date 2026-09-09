"""Robustness to feed uncertainty -- model_spec.md §33, eq (64).

    (64)  max_{d in U} [ S_hat(u, w, d) + z_{1-alpha} sigma_pred ] <= 10
          linear-in-d approximation -> add Gamma * ||grad_d S_hat||

``U`` is a box or an ellipsoid around the nominal feed properties.  For a
model that is affine in ``d`` the worst case has a closed form:

    box (l-inf, half-widths r):  max_d g'(d-d0) = sum_i r_i |g_i|   (dual l-1)
    ellipsoid (d-d0)'P^-1(d-d0) <= 1:  max = sqrt(g' P g)           (dual l-2)

``Gamma`` scales the conservatism (§33).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UncertaintySet:
    """U of eq (64)."""
    center: np.ndarray
    kind: str                      # "box" | "ellipsoid"
    half_widths: np.ndarray | None = None   # box
    shape: np.ndarray | None = None         # ellipsoid P
    names: list[str] | None = None
    status: str = "[DOP]"
    provenance: str = ""

    def __post_init__(self):
        self.center = np.asarray(self.center, float).ravel()
        if self.kind == "box":
            if self.half_widths is None:
                raise ValueError("box set needs half_widths")
            self.half_widths = np.asarray(self.half_widths, float).ravel()
            if self.half_widths.size != self.center.size:
                raise ValueError("half_widths size mismatch")
            if np.any(self.half_widths < 0):
                raise ValueError("half_widths must be non-negative")
        elif self.kind == "ellipsoid":
            if self.shape is None:
                raise ValueError("ellipsoid set needs a shape matrix")
            self.shape = np.atleast_2d(np.asarray(self.shape, float))
            if self.shape.shape[0] != self.center.size:
                raise ValueError("shape matrix size mismatch")
        else:
            raise ValueError("kind must be 'box' or 'ellipsoid'")

    # ------------------------------------------------------------------
    def support(self, grad) -> float:
        """max_{d in U} grad' (d - center)  -- the eq 64 worst-case increment."""
        g = np.asarray(grad, float).ravel()
        if g.size != self.center.size:
            raise ValueError("gradient size mismatch")
        if self.kind == "box":
            return float(np.sum(self.half_widths * np.abs(g)))
        return float(np.sqrt(max(g @ self.shape @ g, 0.0)))

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Draw ``n`` points from U -- used to verify robust feasibility."""
        rng = np.random.default_rng() if rng is None else rng
        if self.kind == "box":
            return self.center + rng.uniform(-1, 1, (n, self.center.size)) \
                * self.half_widths
        L = np.linalg.cholesky(self.shape + 1e-12 * np.eye(self.center.size))
        z = rng.normal(size=(n, self.center.size))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        r = rng.uniform(size=(n, 1)) ** (1.0 / self.center.size)
        return self.center + (z * r) @ L.T

    def contains(self, d) -> bool:
        v = np.asarray(d, float).ravel() - self.center
        if self.kind == "box":
            return bool(np.all(np.abs(v) <= self.half_widths + 1e-12))
        return bool(v @ np.linalg.solve(self.shape, v) <= 1.0 + 1e-9)


def robust_margin(grad_d, U: UncertaintySet, gamma: float = 1.0) -> float:
    """Gamma * ||grad_d S_hat||_U*  -- the additive term of eq (64)."""
    if gamma < 0:
        raise ValueError("Gamma must be non-negative")
    return float(gamma) * U.support(grad_d)


def worst_case_value(y_nominal: float, grad_d, U: UncertaintySet,
                     gamma: float = 1.0) -> float:
    """max_{d in U} y_hat(d) for a model affine in d (eq 64)."""
    return float(y_nominal) + robust_margin(grad_d, U, gamma)


def box_from_history(values, q: float = 0.95, gamma_scale: float = 1.0
                     ) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a box U from the historical spread of the feed properties.

    Appendix A lists ``U`` as [DOP] derived from the spread of ``crude_*``.
    The note records that a historical spread is an estimate of the plausible
    variation, not a certified envelope.
    """
    X = np.atleast_2d(np.asarray(values, float))
    m = np.isfinite(X).all(axis=1)
    if m.sum() < 50:
        raise ValueError("need at least 50 complete rows")
    X = X[m]
    center = np.median(X, axis=0)
    half = gamma_scale * np.quantile(np.abs(X - center), q, axis=0)
    note = (f"[DOP] box U from the {q:.0%} quantile of |d - median| over the "
            "training window; an estimate of plausible feed variation, not a "
            "certified envelope.")
    return center, half, note
