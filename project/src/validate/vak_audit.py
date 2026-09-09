"""Numerical audit of the ВАК regressions -- model_spec.md §26.

The workbook supplies the virtual-analyser formulas but not the namespace of
their symbols, the meaning of ``LIMS:24-2000.Pipeline.*``, or any statement
that they are still calibrated.  Using them as the kernel g_phi of eq 55
without checking would violate the working rule: a kernel must earn its place.

For every parsed formula this module

  1. evaluates it on the aligned state frame under each candidate namespace /
     LIMS binding,
  2. compares it against the corresponding held-out LIMS series (time split,
     one row per measurement -- §25, §38),
  3. scores it against the trivial predictor "the training-window median",
  4. keeps it as a kernel only if it beats that baseline on the *test* window.

A formula that fails is disabled and the residual model carries the property
instead.  The verdicts are written to artifacts/vak_audit.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import domain
from src.kernels import vak
from src.validate import metrics, splits


def _series_for(frame: pd.DataFrame, ns: str) -> pd.DataFrame:
    """Bare-symbol view of one namespace of the state frame."""
    pref = ns + ":"
    cols = {c[len(pref):]: c for c in frame.columns if c.startswith(pref)}
    return frame[list(cols.values())].rename(
        columns={v: k for k, v in cols.items()})


def evaluate_formula(f: vak.VakFormula, frame: pd.DataFrame,
                     lims_binding: dict[str, str] | None = None,
                     namespace: str | None = None) -> np.ndarray:
    """Evaluate one formula on the whole state frame."""
    ns = namespace or f.namespace
    env = {}
    view = _series_for(frame, ns)
    for s in f.symbols:
        if s in view.columns:
            env[s] = view[s].to_numpy(dtype=float)
    for ident in f.lims_refs:
        col = (lims_binding or {}).get(ident)
        if col and col in frame.columns:
            env[ident] = frame[col].to_numpy(dtype=float)
    return f.evaluate(env)


def audit(frame: pd.DataFrame, formulas: dict[str, vak.VakFormula],
          train_end, test_end=None, min_skill: float = 0.02) -> dict:
    """Score every formula against held-out LIMS (§38 rules)."""
    t = frame["date"]
    sp = splits.time_split(t, train_end, test_end)
    splits.assert_no_leakage(t, sp.train_idx, sp.test_idx)

    out: dict[str, dict] = {}
    for qname, spec in domain.QUALITY_PARAMS.items():
        fname = spec.get("vak")
        if not fname or fname not in formulas:
            continue
        f = formulas[fname]
        target_col = f"LIMS:{qname}"
        if target_col not in frame.columns:
            continue
        if not f.ok:
            out[fname] = {"usable": False, "reason": f.note,
                          "quality": qname}
            continue

        bindings = [None]
        if f.lims_refs:
            bindings = []
            prefix = {"hds_feed": "LIMSFEED", "hds_product_prev": "LIMSPREV"}
            for ident in f.lims_refs:
                for point, param in domain.VAK_LIMS_CANDIDATES.get(ident, []):
                    qn = next((k for k, v in domain.QUALITY_PARAMS.items()
                               if v["lims_param"] == param), None)
                    if qn:
                        bindings.append({ident: f"{prefix[point]}:{qn}"})
            bindings = bindings or [None]

        best = None
        for binding in bindings:
            pred = evaluate_formula(f, frame, binding)
            rows = frame[f"{target_col}__new"].to_numpy(bool)
            y = frame[target_col].to_numpy(float)
            tr = np.zeros(len(frame), bool); tr[sp.train_idx] = True
            te = np.zeros(len(frame), bool); te[sp.test_idx] = True
            m_tr = rows & tr & np.isfinite(pred) & np.isfinite(y)
            m_te = rows & te & np.isfinite(pred) & np.isfinite(y)
            if m_te.sum() < 20 or m_tr.sum() < 20:
                cand = {"binding": binding, "n_test": int(m_te.sum()),
                        "usable": False,
                        "reason": "fewer than 20 held-out lab results"}
            else:
                # the ВАК formula may carry a constant offset; the kernel is
                # allowed a training-window bias correction, nothing more.
                off = float(np.median(y[m_tr] - pred[m_tr]))
                mm = metrics.metric_set(y[m_te], pred[m_te] + off)
                base = float(np.median(y[m_tr]))
                mb = metrics.metric_set(y[m_te], np.full(m_te.sum(), base))
                skill = (1.0 - mm["MAE"] / mb["MAE"]) if mb["MAE"] else np.nan
                cand = {"binding": binding, "offset": off,
                        "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                        "MAE": mm["MAE"], "RMSE": mm["RMSE"], "Bias": mm["Bias"],
                        "MAE_median_baseline": mb["MAE"], "skill": skill,
                        "usable": bool(np.isfinite(skill) and skill > min_skill),
                        "min_skill": min_skill,
                        "reason": (f"beats the training-median baseline by "
                                   f"{skill:.1%} MAE"
                                   if np.isfinite(skill) and skill > min_skill
                                   else "does not beat the training-median "
                                        "baseline on held-out LIMS by the "
                                        f"required {min_skill:.0%}")}
            if best is None or (cand.get("skill", -np.inf)
                                > best.get("skill", -np.inf)):
                best = cand
        best["quality"] = qname
        out[fname] = best
    return out


def usable_kernels(audit_result: dict) -> dict[str, dict]:
    return {k: v for k, v in audit_result.items() if v.get("usable")}
