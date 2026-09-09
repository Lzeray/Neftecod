"""Runtime assembly: dataset -> identification -> agents -> one decision cycle.

This is the only place that wires the blocks together.  Every block is still
usable (and tested) on its own; nothing here re-implements model behaviour.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import domain
from src.agents.contracts import PlantState
from src.agents.optimization_agent import OptimizationAgent
from src.agents.orchestrator import Orchestrator
from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.config import Config, load_config
from src.data.dataset import Dataset, build_dataset
from src.ident import channels as CH
from src.ident.pipeline import IdentifiedModel, identify
from src.optimize.robust import UncertaintySet, box_from_history
from src.predict.plant import PlantPredictor, replay
from src.predict.uncertainty import empirical_half_widths
from src.reliability.severity import SeverityIndex
from src.validate import splits


@dataclass
class System:
    cfg: Config
    ds: Dataset
    model: IdentifiedModel
    predictor: PlantPredictor
    utilde: pd.DataFrame
    replay_frame: pd.DataFrame
    channel_dynamics: dict
    orchestrator: Orchestrator
    quality_agent: QualityAgent
    reliability_agent: ReliabilityAgent
    optimization_agent: OptimizationAgent
    reports: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def state_at(self, ts) -> tuple[PlantState, pd.Series, pd.Series, float, float]:
        t = pd.Timestamp(ts)
        f = self.ds.frame
        idx = f.index[f["date"] <= t]
        if len(idx) == 0:
            raise KeyError(f"no telemetry at or before {t}")
        i = int(idx[-1])
        row = f.loc[i]
        ur = self.utilde.loc[i]
        rr = self.replay_frame.loc[i]

        u_now = {k: float(row[k]) for k in domain.MANIPULATED_KEYS
                 if k in f.columns}
        d_now = {k: float(row[k]) for k in domain.DISTURBANCE_KEYS
                 if k in f.columns}
        derived = {k: float(row[k]) for k in f.columns
                   if k.startswith("derived:")}
        if "LIMSFEED:sulfur" in f.columns:
            derived["feed_sulfur_mgkg"] = float(row["LIMSFEED:sulfur"])
        fresh = {}
        for src in ("LIMS:sulfur", "PAK:sulfur", "LIMSFEED:sulfur"):
            col = f"{src}__age_h"
            fresh[src] = float(row[col]) if col in f.columns else float("nan")
        flags = {}
        for col in f.columns:
            if col.endswith("__stale") or col.endswith("__outlier"):
                flags[col] = bool(row[col])
        flags["LIMS:sulfur__missing"] = not np.isfinite(row.get("LIMS:sulfur",
                                                                np.nan))
        # Severity proxies are not all manipulated variables (the R-202
        # differential pressure is a pure measurement), so the raw values of
        # every proxy tag travel with the state; otherwise the reliability
        # agent silently reports them as "missing".
        raw = {"index": i}
        for pr in domain.SEVERITY_PROXIES:
            tag = pr["tag"]
            if tag in f.columns:
                raw[tag] = float(row[tag])
        for extra in ("LIMSFEED:sulfur", "LIMS:sulfur", "PAK:sulfur"):
            if extra in f.columns:
                raw[extra] = float(row[extra])
        state = PlantState(timestamp=str(row["date"]), u_now=u_now, d_now=d_now,
                           derived=derived, freshness=fresh, flags=flags,
                           raw_row=raw)
        return state, row, ur, float(rr["eta"]), float(rr["P_eta"])

    def decide(self, ts, persist: bool = True):
        state, row, ur, eta, P = self.state_at(ts)
        return self.orchestrator.decide(state, row, ur, eta, P, persist=persist)


# ----------------------------------------------------------------------
def build_system(cfg: Config | None = None, use_cache: bool = True) -> System:
    cfg = cfg or load_config()
    ds = build_dataset(cfg, use_cache=use_cache)
    model = _identify_cached(ds, cfg, use_cache)

    Ts = cfg.Ts
    dyn = {k: CH.ChannelDynamics(
        key=k, delay_samples=int(v["delay_samples"]),
        tau_samples=float(v["tau_samples"]), peak_corr=float(v["peak_corr"]),
        n_obs=int(v["n_obs"]), Ts_s=Ts)
        for k, v in model.channel_dynamics.items()}
    utilde = CH.build_effective_inputs(ds.frame, dyn, Ts)
    predictor = PlantPredictor(model.kernel, model.residual_names,
                               np.asarray(model.residual_theta, float),
                               float(model.residual_intercept), list(dyn))
    fp = model.filter_params
    rr = replay(predictor, ds.frame, utilde,
                q_bias_per_h=float(fp["q_bias_per_h"]),
                r_lims=float(fp["r_lims"]),
                r_pak=float(cfg["filter"]["r_pak"]),
                tau_lims_max_h=float(cfg["spec"]["tau_lims_max_h"]),
                tau_pak_max_h=float(cfg["spec"]["tau_pak_max_h"]),
                use_pak=bool(fp["use_pak"]))

    unc = model.uncertainty_model(
        lev_scale=float(cfg["uncertainty"]["lev_scale"]),
        lev_cap=float(cfg["uncertainty"].get("lev_cap", 3.0)))
    ss = model.state_space(Ts)
    # set-based bounds W, V calibrated on the *training* residuals (§28)
    train = _train_mask(ds.frame, cfg)
    lab = ds.frame["LIMS:sulfur__new"].to_numpy(bool)
    resid = (ds.frame["LIMS:sulfur"].to_numpy(float)
             - rr.frame["y_hat"].to_numpy(float))
    m = train & lab & np.isfinite(resid)
    v_half = empirical_half_widths(resid[m], 0.99)
    w_half = np.full(ss.n, 0.0) if ss.n else np.zeros(0)

    qa = QualityAgent(predictor, unc, cfg, state_space=ss if ss.n else None,
                      w_half=w_half if ss.n else None,
                      v_half=np.array([v_half]) if ss.n else None,
                      channel_dynamics=dyn)
    sev_index = SeverityIndex(model.severity_terms)
    ra = ReliabilityAgent(sev_index, model.deactivation, cfg)

    u_set, u_note = _feed_uncertainty(ds.frame, train, cfg)
    oa = OptimizationAgent(model.bounds, cfg, u_set=u_set,
                           severity_index=sev_index)
    pak_usable = bool(fp.get("use_pak", False))
    pak_reason = ("" if pak_usable else
                  "the inner-split search found that letting the stream "
                  "analyser move the offset made the forecast worse (its "
                  "correlation with the laboratory result is ~0.2 at best)")
    orch = Orchestrator(qa, ra, oa, cfg, artifacts_dir=cfg.artifacts_dir,
                        pak_usable=pak_usable, pak_reason=pak_reason)

    reports = dict(model.reports)
    reports["uncertainty_set"] = u_note
    reports["set_based"] = {"v_half_mgkg": v_half, "n_states": int(ss.n)}
    reports["replay"] = {"anchors": rr.n_anchors, "gate_rejected": rr.n_rejected}
    return System(cfg=cfg, ds=ds, model=model, predictor=predictor,
                  utilde=utilde, replay_frame=rr.frame, channel_dynamics=dyn,
                  orchestrator=orch,
                  quality_agent=qa, reliability_agent=ra,
                  optimization_agent=oa, reports=reports)


def _train_mask(frame, cfg) -> np.ndarray:
    sp = splits.time_split(frame["date"], cfg["time"]["train_end"],
                           cfg["time"]["test_end"])
    m = np.zeros(len(frame), bool)
    m[sp.train_idx] = True
    if "derived:operating" in frame.columns:
        m &= frame["derived:operating"].to_numpy(bool)
    return m


def _feed_uncertainty(frame, train, cfg):
    """Box U for the unmeasured/uncertain feed properties (eq 64)."""
    col = "LIMSFEED:sulfur"
    if col not in frame.columns:
        return None, {"available": False}
    vals = frame.loc[train, [col]].to_numpy(float)
    try:
        center, half, note = box_from_history(vals, q=0.95)
    except ValueError:
        return None, {"available": False, "reason": "too few feed lab results"}
    return (UncertaintySet(center=center, kind="box", half_widths=half,
                           names=["feed_sulfur_mgkg"], status="[DOP]",
                           provenance=note),
            {"available": True, "center": center.tolist(),
             "half_widths": half.tolist(), "note": note,
             "unmeasured": domain.UNMEASURED_DISTURBANCES})


def _identify_cached(ds: Dataset, cfg: Config, use_cache: bool) -> IdentifiedModel:
    p = cfg.cache_dir / "identified_model.pkl"
    if use_cache and p.exists():
        try:
            with open(p, "rb") as fh:
                m = pickle.load(fh)
            if m.train_end == str(cfg["time"]["train_end"]):
                return m
        except Exception:                       # noqa: BLE001
            pass
    m = identify(ds, cfg)
    if use_cache:
        with open(p, "wb") as fh:
            pickle.dump(m, fh)
    return m
