"""Identifiability and persistent excitation -- model_spec.md §19, eq (44).

    (44)  rank(Phi) = p                (uniqueness)
          (1/N) Phi^T Phi >= alpha I   (persistent excitation)

§19: regularisation makes the computation stable but does **not** supply
information that is absent from the data.  Channels that fail this test are
removed from the manipulated set (§31) instead of being fitted anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExcitationReport:
    n: int
    p: int
    rank: int
    min_eig: float
    cond: float
    full_rank: bool
    persistently_exciting: bool
    collinear_groups: list[list[int]] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)


def excitation_report(Phi: np.ndarray, alpha: float = 1e-6,
                      cond_max: float = 1e8,
                      corr_max: float = 0.995) -> ExcitationReport:
    """Rank / PE / collinearity diagnosis of a regressor matrix (eq 44)."""
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    n, p = Phi.shape
    if n == 0 or p == 0:
        return ExcitationReport(n, p, 0, 0.0, np.inf, False, False)
    G = Phi.T @ Phi / max(n, 1)
    eig = np.linalg.eigvalsh((G + G.T) / 2.0)
    min_eig = float(eig.min())
    max_eig = float(eig.max())
    cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    rank = int(np.linalg.matrix_rank(Phi))

    groups: list[list[int]] = []
    sd = Phi.std(axis=0)
    live = np.where(sd > 0)[0]
    if live.size >= 2:
        C = np.corrcoef(Phi[:, live], rowvar=False)
        seen: set[int] = set()
        for i in range(live.size):
            if i in seen:
                continue
            grp = [int(live[i])]
            for j in range(i + 1, live.size):
                if abs(C[i, j]) >= corr_max:
                    grp.append(int(live[j]))
                    seen.add(j)
            if len(grp) > 1:
                groups.append(grp)
    return ExcitationReport(
        n=n, p=p, rank=rank, min_eig=min_eig, cond=cond,
        full_rank=rank == p,
        persistently_exciting=min_eig >= alpha and cond <= cond_max,
        collinear_groups=groups,
    )


def prune_collinear(Phi: np.ndarray, names: list[str], alpha: float = 1e-6,
                    cond_max: float = 1e8, corr_max: float = 0.995
                    ) -> tuple[np.ndarray, list[str], ExcitationReport]:
    """Drop the redundant member of each near-collinear group (§19, §31).

    The *first* column of a group is kept (callers order the channels by
    physical preference), the rest are reported as not identifiable.
    """
    rep = excitation_report(Phi, alpha, cond_max, corr_max)
    drop: set[int] = set()
    for grp in rep.collinear_groups:
        drop.update(grp[1:])
    # constant columns carry no information at all
    sd = np.atleast_2d(Phi).std(axis=0)
    drop.update(int(i) for i in np.where(sd <= 0)[0])
    keep = [i for i in range(Phi.shape[1]) if i not in drop]
    rep.dropped = sorted(drop)
    return Phi[:, keep], [names[i] for i in keep], rep
