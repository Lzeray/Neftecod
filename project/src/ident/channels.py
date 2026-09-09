"""Channel dynamics identification -- model_spec.md §12, §37 step 1.

For every manipulated variable the transport delay ``theta_j`` and time
constant ``tau_j`` of eq 32 are estimated from the cross-correlation of the
shaped input against the laboratory target, on the **training window only**.

model_spec.md §12 is explicit that a correlation peak is a *candidate*, not an
established delay, so every result carries the peak correlation and the number
of laboratory results behind it, and channels whose peak is not distinguishable
from noise are reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.lag import first_order_filter, foptd


@dataclass
class ChannelDynamics:
    key: str
    delay_samples: int
    tau_samples: float
    peak_corr: float
    n_obs: int
    Ts_s: float
    status: str = "[DOP]"
    note: str = ""

    @property
    def delay_h(self) -> float:
        return self.delay_samples * self.Ts_s / 3600.0

    @property
    def tau_h(self) -> float:
        return self.tau_samples * self.Ts_s / 3600.0

    @property
    def significant(self) -> bool:
        """|r| above the ~2-sigma noise floor 2/sqrt(n) of a zero correlation."""
        return self.n_obs > 8 and abs(self.peak_corr) > 2.0 / np.sqrt(self.n_obs)

    def as_dict(self) -> dict:
        return {"key": self.key, "delay_samples": self.delay_samples,
                "delay_h": self.delay_h, "tau_samples": self.tau_samples,
                "tau_h": self.tau_h, "peak_corr": self.peak_corr,
                "n_obs": self.n_obs, "significant": self.significant,
                "status": self.status, "note": self.note}


def identify_channel(u: np.ndarray, y: np.ndarray, rows: np.ndarray,
                     Ts_s: float, max_delay: int = 36,
                     tau_grid=(0.0, 3.0, 6.0, 12.0, 24.0, 48.0),
                     robust_clip: float | None = None) -> ChannelDynamics:
    """Grid search over (delay, tau) maximising |corr(u_tilde, y)| on ``rows``.

    ``rows`` is the boolean mask of the laboratory results used for scoring --
    one row per measurement (§25), inside the training window only (§38).
    """
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    uf = pd.Series(u).ffill().bfill().to_numpy()
    yy = y.copy()
    if robust_clip is not None:
        rows = rows & (np.abs(yy) < robust_clip)

    best = (0, 0.0, 0.0)
    n_obs = int(rows.sum())
    idx = np.where(rows & np.isfinite(yy))[0]
    if idx.size < 20:
        return ChannelDynamics(key="", delay_samples=0, tau_samples=0.0,
                               peak_corr=0.0, n_obs=n_obs, Ts_s=Ts_s,
                               note="fewer than 20 laboratory results in the "
                                    "training window; no delay identified")
    yv = yy[idx]
    yc = yv - yv.mean()
    sy = yv.std()
    if sy <= 0:
        return ChannelDynamics(key="", delay_samples=0, tau_samples=0.0,
                               peak_corr=0.0, n_obs=n_obs, Ts_s=Ts_s,
                               note="target has zero variance on these rows")
    for tau_h in tau_grid:
        # A pure delay and a first-order lag are both LTI, so they commute:
        # foptd(u, tau, n*Ts) == shift(first_order_filter(u, tau), n).  The
        # filter is therefore evaluated once per tau instead of once per
        # (tau, delay) pair -- 6 passes over 189k samples instead of 222.
        z = first_order_filter(uf, tau_h * 3600.0, Ts_s)
        for d in range(0, max_delay + 1):
            src = idx - d
            ok = src >= 0
            if ok.sum() < 20:
                continue
            a = z[src[ok]]
            b = yc[ok] if ok.all() else (yv[ok] - yv[ok].mean())
            sa = a.std()
            sb = sy if ok.all() else yv[ok].std()
            if sa <= 0 or sb <= 0:
                continue
            r = float(((a - a.mean()) * b).mean() / (sa * sb))
            if abs(r) > abs(best[2]):
                best = (d, tau_h * 3600.0 / Ts_s, r)
    return ChannelDynamics(key="", delay_samples=best[0], tau_samples=best[1],
                           peak_corr=best[2], n_obs=n_obs, Ts_s=Ts_s,
                           note="Cross-correlation candidate (model_spec.md "
                                "§12): a peak is a hypothesis to be tested by "
                                "the held-out residual, not a measured delay.")


def identify_all(frame: pd.DataFrame, keys: list[str], target: str,
                 rows: np.ndarray, Ts_s: float, max_delay: int = 36,
                 robust_clip: float | None = None) -> dict[str, ChannelDynamics]:
    y = frame[target].to_numpy(float)
    out = {}
    for k in keys:
        if k not in frame.columns:
            continue
        cd = identify_channel(frame[k].to_numpy(float), y, rows, Ts_s,
                              max_delay, robust_clip=robust_clip)
        cd.key = k
        out[k] = cd
    return out


#: Which manipulated channel drives each input of the physical kernel, and so
#: whose identified (delay, time constant) the kernel input must be shaped by.
#: model_spec.md §26: "Все входы -- запаздывающие/сглаженные (u_tilde по
#: ур. 32-34), не мгновенные."  Feeding the kernel the *instantaneous* reactor
#: temperature would claim that a set-point change reaches the product
#: immediately, when identification puts the response 2-3 h later.
KERNEL_INPUT_DRIVERS: dict[str, tuple[str, str]] = {
    # kernel input column      -> (logical name, driving channel)
    "derived:T_WABT_C":        ("T_WABT", "HDS:T5"),
    "derived:pH2_proxy_MPa":   ("pH2", "HDS:P3"),
    "derived:LHSV_rel":        ("LHSV", "HDS:F26"),
    # feed sulfur travels with the feed, so it shares the feed channel's
    # transport delay
    "LIMSFEED:sulfur":         ("S_in", "HDS:F26"),
}

#: Column prefix of the shaped kernel inputs inside the u_tilde frame.
KIN = "kin:"


def build_effective_inputs(frame: pd.DataFrame,
                           dyn: dict[str, ChannelDynamics],
                           Ts_s: float) -> pd.DataFrame:
    """u_tilde for every channel (eq 32-33), plus the shaped kernel inputs.

    The frame carries two families of columns:

      ``<tag>~lag``  -- the effective manipulated inputs of eq 33, used by the
                        data-driven residual r_psi;
      ``kin:<name>`` -- the kernel's own inputs, shaped by the dynamics of the
                        channel that drives them (see KERNEL_INPUT_DRIVERS), so
                        that g_phi also sees delayed values as §26 requires.
    """
    out = pd.DataFrame(index=frame.index)
    for k, d in dyn.items():
        u = pd.Series(frame[k].to_numpy(float)).ffill().bfill().to_numpy()
        out[f"{k}~lag"] = foptd(u, d.tau_samples * Ts_s,
                                d.delay_samples * Ts_s, Ts_s)
    for col, (name, driver) in KERNEL_INPUT_DRIVERS.items():
        if col not in frame.columns:
            continue
        v = pd.Series(frame[col].to_numpy(float)).ffill().bfill().to_numpy()
        d = dyn.get(driver)
        if d is None:
            out[f"{KIN}{name}"] = v
            continue
        out[f"{KIN}{name}"] = foptd(v, d.tau_samples * Ts_s,
                                    d.delay_samples * Ts_s, Ts_s)
    return out


def kernel_input_report(dyn: dict[str, ChannelDynamics]) -> dict:
    """What shaping each kernel input received, for the assumptions register."""
    out = {}
    for col, (name, driver) in KERNEL_INPUT_DRIVERS.items():
        d = dyn.get(driver)
        out[name] = {
            "source_column": col, "driving_channel": driver,
            "delay_h": (d.delay_h if d else 0.0),
            "tau_h": (d.tau_h if d else 0.0),
            "note": "model_spec.md §26: the kernel sees the shaped input of "
                    "eq 32-33, not the instantaneous measurement",
        }
    return out
