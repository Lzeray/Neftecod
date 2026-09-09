"""Blending indices for non-linear properties -- model_spec.md §21, eq (46).

    (46)  BI(y_prod) = sum_i w_i BI(y_i),   y_prod = BI^{-1}( sum_i w_i BI(y_i) )

Sulfur blends linearly in mass (eq 7) and must **not** go through an index.
CFPP / cloud point / flash point / viscosity must **not** be blended linearly
and must go through an index.  ``blending.py`` enforces that split.

The index forms are literature correlations [LIT] with fitted constants
[DOP]; they are declared in the assumptions register and can be recalibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BlendIndex:
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    status: str = "[LIT/DOP]"
    note: str = ""

    def __call__(self, y):
        return self.forward(np.asarray(y, dtype=float))

    def inv(self, bi):
        return self.inverse(np.asarray(bi, dtype=float))


def _k(t_c):
    return np.asarray(t_c, dtype=float) + 273.15


# --- cold-flow properties: the classical power-law index in absolute K ------
_CFPP_EXP = 0.05
_CLOUD_EXP = 0.05


CFPP_INDEX = BlendIndex(
    "CFPP",
    forward=lambda t_c: np.power(_k(t_c), 1.0 / _CFPP_EXP) * 1e-60,
    inverse=lambda bi: np.power(np.maximum(bi, 1e-300) * 1e60, _CFPP_EXP) - 273.15,
    note="[LIT] power-law cold-flow index in absolute temperature; exponent "
         "0.05 is the common refinery value and is a calibration target.",
)

CLOUD_INDEX = BlendIndex(
    "CloudPoint",
    forward=lambda t_c: np.power(_k(t_c), 1.0 / _CLOUD_EXP) * 1e-60,
    inverse=lambda bi: np.power(np.maximum(bi, 1e-300) * 1e60, _CLOUD_EXP) - 273.15,
    note="[LIT] same family as CFPP.",
)

FLASH_INDEX = BlendIndex(
    "FlashPoint",
    forward=lambda t_c: np.power(10.0, -6.1188 + 2414.0 / (_k(t_c) - 42.6)),
    inverse=lambda bi: 2414.0 / (np.log10(np.maximum(bi, 1e-300)) + 6.1188)
                       + 42.6 - 273.15,
    note="[LIT] API/Wickey-Chittenden flash-point blending index.",
)

VISCOSITY_INDEX = BlendIndex(
    "Viscosity",
    forward=lambda nu: np.log(np.log(np.maximum(nu, 1e-6) + 0.7)),
    inverse=lambda bi: np.exp(np.exp(bi)) - 0.7,
    note="[LIT] Refutas / ASTM double-log viscosity blending index.",
)

#: linear-in-mass properties -- no index (eq 7).  Kept explicit so that a caller
#: cannot silently blend a non-linear property linearly.
LINEAR_PROPERTIES = {"Mg.Sulfur", "Mass.Sulfur", "sulfur_mgkg", "D15",
                     "density15", "CetaneNumber"}

INDEXED_PROPERTIES: dict[str, BlendIndex] = {
    "CFPP": CFPP_INDEX,
    "CloudPoint": CLOUD_INDEX,
    "FlashPoint": FLASH_INDEX,
    "Viscosity": VISCOSITY_INDEX,
    "T95": BlendIndex(
        "T95",
        forward=lambda t_c: np.power(_k(t_c), 3.0),
        inverse=lambda bi: np.power(np.maximum(bi, 0.0), 1.0 / 3.0) - 273.15,
        note="[DOP] distillation temperatures are blended through a cubic "
             "index; volumetric-average T95 is not additive.",
    ),
}


def blend_indexed(w, y_components, index: BlendIndex) -> float:
    """y_prod = BI^{-1}( sum_i w_i BI(y_i) )   (eq 46)."""
    w = np.asarray(w, dtype=float)
    y = np.asarray(y_components, dtype=float)
    if w.shape != y.shape:
        raise ValueError("weights and component values must have equal shape")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"blend fractions must sum to 1, got {w.sum()}")
    if np.any(w < -1e-12):
        raise ValueError("blend fractions must be non-negative")
    return float(index.inv(float(np.sum(w * index(y)))))


def get_index(prop: str) -> BlendIndex:
    if prop in INDEXED_PROPERTIES:
        return INDEXED_PROPERTIES[prop]
    raise KeyError(f"{prop!r} has no blending index; is it a linear property?")


# ----------------------------------------------------------------------
def depressant_shift(dose_ppm, dmax_c: float, k_ppm: float) -> np.ndarray:
    """Additive, saturating CFPP shift from a cold-flow-improver dose (§21).

        dCFPP(dose) = -dmax * dose / (k + dose)

    Negative (the additive lowers CFPP), monotone, saturating at ``-dmax``.
    [DOP] ``dmax`` and ``k`` are calibration targets.
    """
    d = np.asarray(dose_ppm, dtype=float)
    if np.any(d < 0):
        raise ValueError("dose must be non-negative")
    if dmax_c < 0 or k_ppm <= 0:
        raise ValueError("dmax must be >= 0 and k > 0")
    return -dmax_c * d / (k_ppm + d)
