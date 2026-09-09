"""Kalman filter with innovation gating -- model_spec.md §23, eq (48)-(50).

    (48)  x_{k|k-1} = A x_{k-1|k-1} + B u_{k-1} + E d_{k-1}
          P_{k|k-1} = A P_{k-1|k-1} A' + Q
    (49)  r_k = y_k - C x_{k|k-1} - D u_k - F d_k
          K_k = P_{k|k-1} C' (C P_{k|k-1} C' + R)^{-1}
    (50)  x_{k|k} = x_{k|k-1} + K_k r_k
          P_{k|k} = (I - K_k C) P_{k|k-1} (I - K_k C)' + K_k R K_k'
                    (Joseph form -- stays symmetric positive definite)

§23: a standard Kalman filter is *not* robust to a failed analyser, so a
measurement is accepted only when its normalised innovation squared passes the
chi^2 gate **and** the source is not flagged stale.  With no measurement the
update reduces to the prediction and P grows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2


@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray
    n_updates: int = 0
    n_rejected: int = 0
    history: list = field(default_factory=list)


def predict(x, P, A, Q, B=None, u=None, E=None, d=None):
    """eq (48)."""
    A = np.atleast_2d(np.asarray(A, float))
    x = np.asarray(x, float).ravel()
    P = np.atleast_2d(np.asarray(P, float))
    xp = A @ x
    if B is not None and u is not None:
        xp = xp + np.atleast_2d(np.asarray(B, float)) @ np.asarray(u, float).ravel()
    if E is not None and d is not None:
        xp = xp + np.atleast_2d(np.asarray(E, float)) @ np.asarray(d, float).ravel()
    Pp = A @ P @ A.T + np.atleast_2d(np.asarray(Q, float))
    return xp, 0.5 * (Pp + Pp.T)


def innovation(y, x_pred, C, D=None, u=None, F=None, d=None):
    """eq (49) first half."""
    C = np.atleast_2d(np.asarray(C, float))
    yhat = C @ np.asarray(x_pred, float).ravel()
    if D is not None and u is not None:
        yhat = yhat + np.atleast_2d(np.asarray(D, float)) @ np.asarray(u, float).ravel()
    if F is not None and d is not None:
        yhat = yhat + np.atleast_2d(np.asarray(F, float)) @ np.asarray(d, float).ravel()
    return np.asarray(y, float).ravel() - yhat


def nis(r, S) -> float:
    """Normalised innovation squared r' S^{-1} r (the chi^2 gate statistic)."""
    r = np.asarray(r, float).ravel()
    S = np.atleast_2d(np.asarray(S, float))
    return float(r @ np.linalg.solve(S, r))


def chi2_threshold(dof: int, p: float = 0.99) -> float:
    return float(chi2.ppf(p, dof))


def update(x_pred, P_pred, y, C, R, D=None, u=None, F=None, d=None,
           gate_p: float | None = 0.99, stale: bool = False):
    """eq (49)-(50) with the §23 gate.

    Returns ``(x, P, info)``.  ``info["accepted"]`` is False when the source is
    stale or the innovation fails the chi^2 gate, in which case the prediction
    is returned unchanged -- the filter degrades to open-loop prediction rather
    than absorbing a broken analyser.
    """
    x_pred = np.asarray(x_pred, float).ravel()
    P_pred = np.atleast_2d(np.asarray(P_pred, float))
    C = np.atleast_2d(np.asarray(C, float))
    R = np.atleast_2d(np.asarray(R, float))
    y = np.asarray(y, float).ravel()

    info = {"accepted": False, "reason": "", "nis": np.nan, "threshold": np.nan}
    if stale:
        info["reason"] = "source flagged stale (§23)"
        return x_pred, P_pred, info
    if not np.all(np.isfinite(y)):
        info["reason"] = "no measurement -> pure prediction (eq 48)"
        return x_pred, P_pred, info

    S = C @ P_pred @ C.T + R
    r = innovation(y, x_pred, C, D, u, F, d)
    info["nis"] = nis(r, S)
    if gate_p is not None:
        info["threshold"] = chi2_threshold(y.size, gate_p)
        if info["nis"] > info["threshold"]:
            info["reason"] = (f"innovation gate: NIS {info['nis']:.2f} > "
                              f"chi2 {info['threshold']:.2f}")
            return x_pred, P_pred, info

    K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
    x = x_pred + K @ r
    IKC = np.eye(P_pred.shape[0]) - K @ C
    P = IKC @ P_pred @ IKC.T + K @ R @ K.T
    info.update(accepted=True, reason="accepted", gain=K)
    return x, 0.5 * (P + P.T), info


def run_filter(A, B, C, Q, R, ys, us=None, ds=None, E=None, D=None, F=None,
               x0=None, P0=None, gate_p: float | None = 0.99,
               stale_flags=None) -> KalmanState:
    """Convenience loop over eq 48-50; ``ys[k] = None`` means "no measurement"."""
    A = np.atleast_2d(np.asarray(A, float))
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).ravel()
    P = np.eye(n) if P0 is None else np.atleast_2d(np.asarray(P0, float))
    st = KalmanState(x=x, P=P)
    N = len(ys)
    for k in range(N):
        u = None if us is None else us[k]
        d = None if ds is None else ds[k]
        x, P = predict(st.x, st.P, A, Q, B, u, E, d)
        yk = ys[k]
        stale = bool(stale_flags[k]) if stale_flags is not None else False
        if yk is None:
            info = {"accepted": False, "reason": "no measurement"}
        else:
            x, P, info = update(x, P, yk, C, R, D, u, F, d, gate_p, stale)
        st.x, st.P = x, P
        st.n_updates += int(info.get("accepted", False))
        st.n_rejected += int(yk is not None and not info.get("accepted", False))
        st.history.append({"k": k, "x": x.copy(), "P": P.copy(), **info})
    return st
