"""Offset-free augmented model -- model_spec.md §22, eq (47).

    (47)  x_{k+1} = A x_k + B u_k + E d_k + B_eta eta_k + w_k
          eta_{k+1} = eta_k + xi_k
          y_k = C x_k + D u_k + F d_k + C_eta eta_k + v_k

§22: eta is an *offset estimate*, never a measured catalyst activity, and the
augmented system must stay observable -- there cannot be more independent
offsets than the measurements can support.  ``check_observability`` enforces
exactly that and refuses to build an unobservable augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentedModel:
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    C: np.ndarray
    D: np.ndarray
    F: np.ndarray
    n_x: int
    n_eta: int

    @property
    def n(self) -> int:
        return self.n_x + self.n_eta


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    A = np.atleast_2d(A); C = np.atleast_2d(C)
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])


def is_observable(A, C, tol: float = 1e-10) -> bool:
    O = observability_matrix(A, C)
    return int(np.linalg.matrix_rank(O, tol=tol)) == np.atleast_2d(A).shape[0]


def augment_offset(A, B, E, C, D, F, B_eta=None, C_eta=None,
                   strict: bool = True) -> AugmentedModel:
    """Build the augmented system of eq (47).

    ``strict=True`` raises when the augmented pair (A_aug, C_aug) is not
    observable -- the Pannocchia-Rawlings condition that §22 demands.
    """
    A = np.atleast_2d(np.asarray(A, float))
    C = np.atleast_2d(np.asarray(C, float))
    n_x, n_y = A.shape[0], C.shape[0]
    B = np.atleast_2d(np.asarray(B, float)).reshape(n_x, -1)
    E = np.atleast_2d(np.asarray(E, float)).reshape(n_x, -1)
    D = np.atleast_2d(np.asarray(D, float)).reshape(n_y, -1)
    F = np.atleast_2d(np.asarray(F, float)).reshape(n_y, -1)

    C_eta = np.eye(n_y) if C_eta is None else np.atleast_2d(np.asarray(C_eta, float))
    n_eta = C_eta.shape[1]
    B_eta = (np.zeros((n_x, n_eta)) if B_eta is None
             else np.atleast_2d(np.asarray(B_eta, float)).reshape(n_x, n_eta))

    if n_eta > n_y:
        raise ValueError(
            f"{n_eta} independent offsets cannot be estimated from {n_y} "
            "measurements (model_spec.md §22)")

    A_aug = np.block([[A, B_eta],
                      [np.zeros((n_eta, n_x)), np.eye(n_eta)]])
    C_aug = np.hstack([C, C_eta])
    B_aug = np.vstack([B, np.zeros((n_eta, B.shape[1]))])
    E_aug = np.vstack([E, np.zeros((n_eta, E.shape[1]))])
    if strict and not is_observable(A_aug, C_aug):
        raise ValueError("augmented (A, C) is not observable; reduce the "
                         "number of independent offsets (model_spec.md §22)")
    return AugmentedModel(A_aug, B_aug, E_aug, C_aug, D, F, n_x, n_eta)
