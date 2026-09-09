"""State-space realisation of the identified channels -- eq (30)-(34), (56)-(57).

Each manipulated channel is a first-order lag with a transport delay (eq 32-33).
Written as states -- the filter state plus the delay line of eq 34 -- the whole
predictor becomes the linear system of eq 30, so the multi-step prediction of
eq 56 is exactly the affine map of eq 57 and the optimiser of §32 is a QP.

    per channel j with delay n_j and pole a_j:
        zeta^{(1..n_j+1)}  delay line   (eq 34)
        z_j                filter state (eq 33)
        z_{j,k+1} = a_j z_{j,k} + (1 - a_j) u_{j,k-n_j}

The delay line carries n_j + 1 stages, not n_j: eq 33 feeds ``u_{k-n_j}`` into
the update that produces ``z_{k+1}``, so the input reaching the state at step k
is ``u_{k-n_j-1}``.  That is exactly the strictly causal ZOH numerator
``[0, K(1-a)]`` that scipy.signal.cont2discrete returns, and it makes this
realisation agree with features/lag.foptd sample for sample -- which matters,
because identification uses foptd and prediction uses this realisation.
    output:
        y_k = sum_j c_j z_{j,k}  (+ kernel + offset, added outside)

The realisation is verified against a direct simulation of eq 33 in
tests/test_statespace.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChannelSpec:
    key: str
    delay: int          # n_j, samples
    a: float            # exp(-Ts/tau_j)
    gain: float         # c_j, output units per input unit


@dataclass
class ChannelStateSpace:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    channels: list[ChannelSpec]
    index: dict[str, dict] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.B.shape[1]

    def steady_state(self, u_level: dict[str, float]) -> np.ndarray:
        """State reached by holding every input at a constant level."""
        x = np.zeros(self.n)
        for ch in self.channels:
            v = float(u_level.get(ch.key, 0.0))
            if not np.isfinite(v):
                v = 0.0
            ix = self.index[ch.key]
            for st in ix["delay"]:
                x[st] = v
            x[ix["filter"]] = v
        return x

    def initial_state(self, u_history: dict[str, np.ndarray]) -> np.ndarray:
        """State at the end of the supplied input history.

        The channel is started from the steady state of the oldest available
        sample and driven through the history, which is exact once the history
        is longer than the delay plus a few time constants and degrades
        gracefully (to the steady state) when it is not.  Non-finite samples
        are forward-filled so a gap in the history cannot poison the state.
        """
        first = {}
        hists = {}
        for ch in self.channels:
            h = np.asarray(u_history.get(ch.key, []), float).ravel()
            h = h[np.isfinite(h)] if h.size else h
            hists[ch.key] = h
            first[ch.key] = float(h[0]) if h.size else 0.0
        x = self.steady_state(first)
        n = max((h.size for h in hists.values()), default=0)
        if n <= 1:
            return x
        for k in range(1, n):
            u = np.array([hists[ch.key][min(k, hists[ch.key].size - 1)]
                          if hists[ch.key].size else 0.0
                          for ch in self.channels], dtype=float)
            x = self.A @ x + self.B @ u
        return x

    def simulate(self, x0, U) -> np.ndarray:
        """y_k for k = 1..H given the stacked future inputs U (H, nu)."""
        U = np.atleast_2d(np.asarray(U, float))
        x = np.asarray(x0, float).ravel().copy()
        out = np.zeros(U.shape[0])
        for k in range(U.shape[0]):
            x = self.A @ x + self.B @ U[k]
            out[k] = float((self.C @ x + self.D @ U[k])[0])
        return out


def build_state_space(channels: list[ChannelSpec]) -> ChannelStateSpace:
    """Assemble (A, B, C, D) from the per-channel FOPTD descriptions."""
    index, n = {}, 0
    for ch in channels:
        if ch.delay < 0:
            raise ValueError(f"{ch.key}: delay must be non-negative")
        if not (0.0 <= ch.a < 1.0):
            raise ValueError(f"{ch.key}: pole a must be in [0, 1)")
        d_idx = list(range(n, n + ch.delay + 1))
        n += ch.delay + 1
        f_idx = n
        n += 1
        index[ch.key] = {"delay": d_idx, "filter": f_idx}
    nu = len(channels)
    A = np.zeros((n, n))
    B = np.zeros((n, nu))
    C = np.zeros((1, n))
    D = np.zeros((1, nu))
    for j, ch in enumerate(channels):
        ix = index[ch.key]
        # delay line (eq 34): zeta^(1)_{k+1} = u_k, zeta^(i)_{k+1} = zeta^(i-1)_k
        line = ix["delay"]
        B[line[0], j] = 1.0
        for i in range(1, len(line)):
            A[line[i], line[i - 1]] = 1.0
        # filter state (eq 33), fed by u_{k-n_j-1} = zeta^(n_j+1)_k
        f = ix["filter"]
        A[f, f] = ch.a
        A[f, line[-1]] = 1.0 - ch.a
        C[0, f] = ch.gain
    return ChannelStateSpace(A, B, C, D, list(channels), index)
