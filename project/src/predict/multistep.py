"""Multi-step prediction as an affine map -- model_spec.md §27, eq (56)-(57).

    (56)  x_{k+h|k} = A^h x_{k|k} + sum_{i=0}^{h-1} A^{h-1-i} (B u_{k+i} + E d_{k+i})
          y_{k+h|k} = C x_{k+h|k} + D u_{k+h} + F d_{k+h}
    (57)  Y_hat = O x_{k|k} + T_u U + T_d D_future

O is built from C A^h; T_u from C A^i B plus the direct feed-through D; T_d
likewise from C A^i E and F.  The affine form is what makes the optimiser of
§32 a QP in U.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionMaps:
    O: np.ndarray        # (H*ny, n)
    Tu: np.ndarray       # (H*ny, H*nu)
    Td: np.ndarray       # (H*ny, H*nd)
    H: int
    ny: int
    nu: int
    nd: int

    def predict(self, x0, U, Dfut) -> np.ndarray:
        """Y_hat = O x + T_u U + T_d D  (eq 57), stacked (H*ny,)."""
        x0 = np.asarray(x0, float).ravel()
        U = np.asarray(U, float).ravel()
        Dfut = np.asarray(Dfut, float).ravel()
        if U.size != self.Tu.shape[1]:
            raise ValueError(f"U must have {self.Tu.shape[1]} entries")
        if Dfut.size != self.Td.shape[1]:
            raise ValueError(f"D must have {self.Td.shape[1]} entries")
        return self.O @ x0 + self.Tu @ U + self.Td @ Dfut

    def reshape(self, Y) -> np.ndarray:
        """Stacked Y -> (H, ny)."""
        return np.asarray(Y, float).reshape(self.H, self.ny)


def build_maps(A, B, C, E=None, D=None, F=None, H: int = 10) -> PredictionMaps:
    """Assemble O, T_u, T_d of eq (57).

    Convention (eq 56): ``u_{k+i}`` for i = 0..H-1 drives the state, and the
    output at step h also sees ``u_{k+h}`` through D.  Index h runs 1..H.
    """
    A = np.atleast_2d(np.asarray(A, float))
    C = np.atleast_2d(np.asarray(C, float))
    n, ny = A.shape[0], C.shape[0]
    B = np.zeros((n, 0)) if B is None else np.atleast_2d(np.asarray(B, float)).reshape(n, -1)
    E = np.zeros((n, 0)) if E is None else np.atleast_2d(np.asarray(E, float)).reshape(n, -1)
    nu, nd = B.shape[1], E.shape[1]
    D = np.zeros((ny, nu)) if D is None else np.atleast_2d(np.asarray(D, float)).reshape(ny, nu)
    F = np.zeros((ny, nd)) if F is None else np.atleast_2d(np.asarray(F, float)).reshape(ny, nd)
    if H < 1:
        raise ValueError("horizon must be >= 1")

    Apow = [np.eye(n)]
    for _ in range(H):
        Apow.append(Apow[-1] @ A)

    O = np.zeros((H * ny, n))
    Tu = np.zeros((H * ny, H * nu))
    Td = np.zeros((H * ny, H * nd))
    for h in range(1, H + 1):
        r = slice((h - 1) * ny, h * ny)
        O[r, :] = C @ Apow[h]
        for i in range(h):                 # u_{k+i}, i = 0..h-1
            blk = C @ Apow[h - 1 - i] @ B
            Tu[r, i * nu:(i + 1) * nu] += blk
            blk_d = C @ Apow[h - 1 - i] @ E
            Td[r, i * nd:(i + 1) * nd] += blk_d
        if h - 1 < H:                      # direct feed-through of u_{k+h-1}
            Tu[r, (h - 1) * nu:h * nu] += D
            Td[r, (h - 1) * nd:h * nd] += F
    return PredictionMaps(O, Tu, Td, H, ny, nu, nd)


def recursive_predict(A, B, C, x0, U, Dfut=None, E=None, D=None, F=None,
                      H: int = 10) -> np.ndarray:
    """Step-by-step evaluation of eq (56) -- the reference for the maps."""
    A = np.atleast_2d(np.asarray(A, float))
    C = np.atleast_2d(np.asarray(C, float))
    n, ny = A.shape[0], C.shape[0]
    B = np.zeros((n, 0)) if B is None else np.atleast_2d(np.asarray(B, float)).reshape(n, -1)
    E = np.zeros((n, 0)) if E is None else np.atleast_2d(np.asarray(E, float)).reshape(n, -1)
    nu, nd = B.shape[1], E.shape[1]
    D = np.zeros((ny, nu)) if D is None else np.atleast_2d(np.asarray(D, float)).reshape(ny, nu)
    F = np.zeros((ny, nd)) if F is None else np.atleast_2d(np.asarray(F, float)).reshape(ny, nd)
    U = np.asarray(U, float).reshape(H, nu) if nu else np.zeros((H, 0))
    Dm = (np.zeros((H, nd)) if Dfut is None
          else np.asarray(Dfut, float).reshape(H, nd))
    x = np.asarray(x0, float).ravel().copy()
    out = np.zeros((H, ny))
    for h in range(1, H + 1):
        x = A @ x + (B @ U[h - 1] if nu else 0.0) + (E @ Dm[h - 1] if nd else 0.0)
        out[h - 1] = C @ x + (D @ U[h - 1] if nu else 0.0) \
            + (F @ Dm[h - 1] if nd else 0.0)
    return out.ravel()
