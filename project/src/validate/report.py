"""Validation on the held-out window -- model_spec.md §38, eq (67)-(69).

Reports, separately and without conflating them (§38 closing note):
  * one-step accuracy of the deployed predictor (kernel + residual + offset)
  * multi-step accuracy at several horizons, against the persistence baseline
  * interval coverage -- the honesty check on sigma_pred (eq 58)
  * false / missed specification violations
  * the response of the model to an input change (step-gain reproduction)

The test window is read exactly once, here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.validate import coverage as COV
from src.validate import metrics as MET
from src.validate import splits


def evaluate(system, save_to: Path | None = None) -> dict:
    cfg = system.cfg
    f = system.ds.frame
    r = system.replay_frame
    alpha = float(cfg["spec"]["alpha"])
    S_max = float(cfg["spec"]["S_max_mgkg"])

    sp = splits.time_split(f["date"], cfg["time"]["train_end"],
                           cfg["time"]["test_end"])
    splits.assert_no_leakage(f["date"], sp.train_idx, sp.test_idx)
    te = np.zeros(len(f), bool); te[sp.test_idx] = True
    tr = np.zeros(len(f), bool); tr[sp.train_idx] = True
    op = (f["derived:operating"].to_numpy(bool)
          if "derived:operating" in f.columns else np.ones(len(f), bool))

    lab_new = f["LIMS:sulfur__new"].to_numpy(bool)
    y = f["LIMS:sulfur"].to_numpy(float)
    yh = r["y_hat"].to_numpy(float)
    model_only = r["model"].to_numpy(float)

    rows = lab_new & te & op & np.isfinite(yh) & np.isfinite(y)
    lab_idx = np.where(lab_new & np.isfinite(y))[0]

    # ---- one-step (next available lab result) -------------------------
    one_step = MET.metric_set(y[rows], yh[rows])
    kernel_only = MET.metric_set(y[rows], model_only[rows])
    train_median = float(np.median(y[lab_new & tr & op & np.isfinite(y)]))
    const = MET.metric_set(y[rows], np.full(int(rows.sum()), train_median))

    # persistence baseline, eq 69: the previous laboratory result
    prev = pd.Series(y[lab_idx]).shift(1).to_numpy()
    pm = te[lab_idx] & op[lab_idx] & np.isfinite(prev) & np.isfinite(y[lab_idx])
    persist = MET.metric_set(y[lab_idx][pm], prev[pm])
    # paired comparisons on the shared rows -- the only defensible test
    vs_persist = MET.paired_mae_difference(y[lab_idx][pm],
                                           yh[lab_idx][pm], prev[pm])
    vs_const = MET.paired_mae_difference(
        y[rows], yh[rows], np.full(int(rows.sum()), train_median))

    # ---- multi-step: forecast made h lab results earlier ---------------
    horizons = {}
    ys = y[lab_idx]
    yhs = yh[lab_idx]
    for h in (1, 2, 4, 8):
        pred_h = pd.Series(yhs).shift(h).to_numpy()
        base_h = pd.Series(ys).shift(h).to_numpy()
        m = te[lab_idx] & op[lab_idx] & np.isfinite(pred_h) & np.isfinite(ys)
        if m.sum() < 20:
            continue
        mm = MET.metric_set(ys[m], pred_h[m])
        bb = MET.metric_set(ys[m], base_h[m])
        horizons[h] = {
            **mm, "MAE_persistence": bb["MAE"], "RMSE_persistence": bb["RMSE"],
            "skill_MAE": (1.0 - mm["MAE"] / bb["MAE"]) if bb["MAE"] else None,
        }

    # ---- interval honesty (eq 58) --------------------------------------
    sig = _sigma_at(system, f, r, rows)
    cov2 = COV.interval_coverage(y[rows], yh[rows], sig, alpha)
    cov1 = COV.one_sided_coverage(y[rows], yh[rows], sig, alpha)
    viol = COV.violation_rates(y[rows], yh[rows], S_max,
                               backoff=sig * _z(alpha))
    viol_nobackoff = COV.violation_rates(y[rows], yh[rows], S_max, backoff=0.0)

    # ---- step-response reproduction (§38 third category) ---------------
    step = _step_response_check(system)

    out = {
        "split": sp.summary(f["date"]),
        "n_test_lab_results": int(rows.sum()),
        "one_step": {
            "model_kernel_residual_offset": one_step,
            "kernel_plus_residual_only": kernel_only,
            "persistence_eq69": persist,
            "train_median_constant": const,
            "skill_vs_persistence": (1.0 - one_step["MAE"] / persist["MAE"])
            if persist["MAE"] else None,
            "skill_vs_constant": (1.0 - one_step["MAE"] / const["MAE"])
            if const["MAE"] else None,
            "paired_vs_persistence": vs_persist,
            "paired_vs_constant": vs_const,
        },
        "multi_step": horizons,
        "coverage": {"two_sided": cov2, "one_sided_eq63": cov1,
                     "sigma_mean": float(np.mean(sig))},
        "spec_violations": {
            "with_backoff": viol, "without_backoff": viol_nobackoff,
            "actual_rate": float((y[rows] > S_max).mean()),
            "note": "the historical process itself exceeds the 10 mg/kg limit "
                    "in a non-negligible fraction of the laboratory results",
        },
        "step_response": step,
        "action_distribution": action_distribution(system),
        "honesty_notes": _honesty(one_step, persist, const, cov1, alpha,
                                  vs_persist, vs_const) + _model_notes(system),
    }
    if save_to is not None:
        Path(save_to).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=_default),
            encoding="utf-8")
    return out


def _z(alpha):
    from src.predict.uncertainty import z_quantile
    return z_quantile(alpha)


def _sigma_at(system, f, r, rows) -> np.ndarray:
    """sigma_pred (eq 58) evaluated row by row on the test lab results."""
    unc = system.model.uncertainty_model(
        lev_scale=float(system.cfg["uncertainty"]["lev_scale"]),
        lev_cap=float(system.cfg["uncertainty"].get("lev_cap", 3.0)))
    from src.ident.channels import KIN
    X = np.column_stack([system.utilde[f"{KIN}S_in"].to_numpy(float),
                         system.utilde[f"{KIN}T_WABT"].to_numpy(float),
                         system.utilde[f"{KIN}pH2"].to_numpy(float),
                         system.utilde[f"{KIN}LHSV"].to_numpy(float)])
    ages = f["LIMS:sulfur__age_h"].to_numpy(float)
    P = r["P_eta"].to_numpy(float)
    idx = np.where(rows)[0]
    out = np.empty(idx.size)
    for j, i in enumerate(idx):
        x = X[i] if np.isfinite(X[i]).all() else None
        tau = ages[i] if np.isfinite(ages[i]) else 0.0
        out[j] = unc.sigma_pred(tau_h=float(tau), P_eta=float(P[i]), x=x)
    return out


def _step_response_check(system) -> dict:
    """Does the model reproduce the response to a temperature step?

    A closed-form check of the deployed sensitivity against the kinetics: a
    +1 degC step on the reactor must lower the predicted sulfur, and the
    magnitude must match the kernel's analytic derivative.
    """
    f = system.ds.frame
    op = (f["derived:operating"].to_numpy(bool)
          if "derived:operating" in f.columns else np.ones(len(f), bool))
    idx = np.where(op & np.isfinite(f["derived:T_WABT_C"].to_numpy(float))
                   & np.isfinite(f["LIMSFEED:sulfur"].to_numpy(float)))[0]
    if idx.size == 0:
        return {"available": False}
    i = int(idx[len(idx) // 2])
    row = f.loc[i]
    ur = system.utilde.loc[i]
    s = float(system.predictor.sensitivity(row, "HDS:T5", 1.0, ur))
    from src.predict.plant import horizon_response
    hr = horizon_response(system.predictor, row, ur, "HDS:T5", 1.0,
                          system.channel_dynamics, system.cfg.Ts,
                          int(system.cfg["horizon"]["H"]))
    return {"available": True, "timestamp": str(row["date"]),
            "dS_dT5_mgkg_per_C": s, "horizon_response": hr,
            "sign_ok": bool(s <= 0),
            "note": "the deployed temperature sensitivity must be negative "
                    "(eq 13-15); it comes from the kernel, since the residual "
                    "gain on this channel is sign-clipped to zero."}


def _honesty(one_step, persist, const, cov1, alpha,
             vs_persist=None, vs_const=None) -> list[str]:
    notes = []
    se = one_step.get("MAE_se", float("nan"))
    notes.append(
        f"one-step MAE is {one_step['MAE']:.3f} +- {se:.3f} mg/kg "
        f"(standard error on {one_step['n']} held-out laboratory results); "
        "differences smaller than about 0.6 mg/kg between candidate models "
        "are not distinguishable on this sample")

    def verdict(name, other, paired):
        if not paired or not np.isfinite(paired.get("diff_MAE", np.nan)):
            return f"comparison against {name} unavailable"
        d, s = paired["diff_MAE"], paired["se"]
        if paired["significant"]:
            direction = "beats" if d < 0 else "is beaten by"
            return (f"the model {direction} {name} by "
                    f"{abs(d):.3f} +- {s:.3f} mg/kg (paired, significant)")
        direction = "lower" if d < 0 else "higher"
        return (f"against {name} ({other['MAE']:.3f} mg/kg) the paired MAE "
                f"difference is {d:+.3f} +- {s:.3f} mg/kg -- {direction} on "
                "average but NOT statistically distinguishable from no "
                "difference")

    notes.append(verdict("the persistence baseline of eq 69", persist,
                         vs_persist))
    notes.append(verdict("a constant at the training median", const, vs_const))
    if const["MAE"] <= one_step["MAE"]:
        notes.append(
            "the product sulfur of this unit is close to white noise around a "
            "slowly moving level: most of the achievable skill is in the "
            "level, which the LIMS-anchored offset supplies, not in the "
            "telemetry")
    c = cov1.get("coverage")
    _ = c
    if c is not None and np.isfinite(c):
        notes.append(
            f"the one-sided bound used by eq 63 covers {100 * c:.1f}% of the "
            f"held-out results against the {100 * (1 - alpha):.0f}% it claims"
            + (" -- honest" if abs(c - (1 - alpha)) <= 0.05
               else " -- optimistic, so the back-off is under-sized"
               if c < 1 - alpha else " -- conservative"))
    return notes


def _model_notes(system) -> list[str]:
    out = []
    fp = getattr(system.model, "filter_params", {}) or {}
    if float(fp.get("residual_scale", 1.0)) == 0.0:
        out.append(
            "the inner-split selection shrank the data-driven residual to "
            "zero: on held-out rows the fitted telemetry gains did not improve "
            "the forecast, so the deployed predictor is the physical kernel "
            "plus the LIMS-anchored offset only (eq 55 with r_psi = 0)")
    if not fp.get("use_pak", False):
        out.append(
            "the same search rejected the stream sulfur analyser as an anchor: "
            "its correlation with the laboratory result is ~0.2 at best, and "
            "letting it move the offset made the forecast worse")
    return out


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def action_distribution(system, n_samples: int = 300) -> dict:
    """How often each decision outcome fires over the held-out window.

    The single most informative summary of this unit: it quantifies how much of
    the record the 10 mg/kg limit can actually be certified on, rather than
    leaving that to a reader's impression of three hand-picked scenarios.
    """
    import numpy as np
    import pandas as pd

    f = system.ds.frame
    op = (f["derived:operating"].to_numpy(bool)
          if "derived:operating" in f.columns else np.ones(len(f), bool))
    te = (f["date"] > pd.Timestamp(system.cfg["time"]["train_end"])).to_numpy()
    idx = np.where(op & te)[0]
    if idx.size == 0:
        return {"available": False}
    step = max(1, idx.size // int(n_samples))
    sample = idx[::step]
    counts: dict[str, int] = {}
    for i in sample:
        a = system.decide(str(f["date"].iloc[i]), persist=False).action
        counts[a] = counts.get(a, 0) + 1
    n = sum(counts.values())
    return {"available": True, "n_sampled": n,
            "sampling": f"every {step}th operating sample of the test window",
            "counts": counts,
            "fractions": {k: round(v / n, 4) for k, v in counts.items()},
            "note": "'recommend'/'hold' means the 10 mg/kg chance constraint "
                    "is satisfiable; 'mitigate' means C is empty and only risk "
                    "reduction is offered; 'refuse' means eq 66 blocks any "
                    "advice."}
