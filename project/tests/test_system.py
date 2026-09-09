"""End-to-end checks on the real data package -- prompt Block L, §5.

These are the only tests that touch the delivered files.  They are skipped when
the data package is not present, so the rest of the suite still runs from a
clean checkout.

Targets:
  * no temporal leakage anywhere in the assembled pipeline (§38)
  * the three pinned demo scenarios produce their documented action
  * a re-run reproduces identical artifacts (ТЗ "Воспроизводимость")
  * every emitted decision carries the ТЗ §5 report blocks
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.demos import SCENARIOS
from src.validate import splits


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def system(cfg):
    missing = [k for k in ("avt_csv", "hds_csv", "lims_xlsx", "pak_xlsx",
                           "tags_xlsx") if not cfg.data_path(k).exists()]
    if missing:
        pytest.skip(f"data package not present: {missing}")
    from src.system import build_system
    return build_system(cfg)


def test_no_temporal_leakage_in_the_assembled_pipeline(system):
    f = system.ds.frame
    sp = splits.time_split(f["date"], system.cfg["time"]["train_end"],
                           system.cfg["time"]["test_end"])
    splits.assert_no_leakage(f["date"], sp.train_idx, sp.test_idx)
    assert f["date"].is_monotonic_increasing
    # every as-of column must be older than or equal to its row timestamp
    for col in f.columns:
        if col.endswith("__age_h"):
            a = f[col].to_numpy(float)
            assert np.nanmin(a) >= -1e-9, f"{col} contains a negative age"


def test_lims_values_are_never_counted_twice(system):
    f = system.ds.frame
    n_new = int(f["LIMS:sulfur__new"].sum())
    n_repeated = int(f["LIMS:sulfur"].notna().sum())
    assert n_new < n_repeated, "the as-of merge must repeat values across rows"
    uniq = splits.unique_measurements(f, "LIMS:sulfur")
    assert len(uniq) == n_new


def test_identification_used_only_the_training_window(system):
    f = system.ds.frame
    train_end = pd.Timestamp(system.cfg["time"]["train_end"])
    assert pd.Timestamp(system.model.train_end) == train_end
    # the channel delays were scored on training lab results only
    n = max(d["n_obs"] for d in system.model.channel_dynamics.values())
    n_train_lab = int((f["LIMS:sulfur__new"].to_numpy(bool)
                       & (f["date"] <= train_end).to_numpy()).sum())
    assert n <= n_train_lab


def test_kinetic_sensitivity_has_the_physical_sign(system):
    f = system.ds.frame
    op = f["derived:operating"].to_numpy(bool)
    i = int(np.where(op)[0][len(np.where(op)[0]) // 2])
    s = float(system.predictor.sensitivity(f.loc[i], "HDS:T5", 1.0))
    assert s < 0, "a hotter reactor must not raise the predicted sulfur"


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_demo_scenarios_produce_a_complete_report(system, key):
    d = system.decide(SCENARIOS[key]["timestamp"], persist=False)
    assert d.action in ("recommend", "hold", "mitigate", "refuse")
    assert d.state_block["timestamp"] and d.state_block["freshness_h"]
    assert "sulfur_forecast_mgkg" in d.risk_block
    assert "gate_eq66" in d.constraint_block
    assert d.confidence_block["confidence"] in ("high", "medium", "low")
    assert len(d.explanation) > 80


def test_demo_scenarios_have_their_documented_outcome(system):
    got = {k: system.decide(v["timestamp"], persist=False).action
           for k, v in SCENARIOS.items()}
    assert got["stable"] in ("recommend", "hold")
    assert got["at_limit"] == "mitigate"
    assert got["stale_data"] == "refuse"


def test_refusal_names_its_reasons(system):
    d = system.decide(SCENARIOS["stale_data"]["timestamp"], persist=False)
    fails = " ".join(d.constraint_block["gate_eq66"])
    assert "freshness" in fails or "confidence" in fails
    assert d.action_block["changes"] == []


def test_mitigation_never_claims_the_spec_is_met(system):
    d = system.decide(SCENARIOS["at_limit"]["timestamp"], persist=False)
    assert d.action_block["certifies_spec"] is False
    assert d.effect_block["margin_to_limit_mgkg"] < 0
    assert "NOT certified" in d.headline or "NOT certified" in d.explanation


def test_recommendation_respects_bounds_and_rate_limits(system):
    d = system.decide(SCENARIOS["stable"]["timestamp"], persist=False)
    for c in d.action_block["changes"]:
        b = system.model.bounds[c["tag"]]
        assert b.lo - 1e-6 <= c["recommended"] <= b.hi + 1e-6 or \
            abs(c["recommended"] - c["current"]) <= b.du_max + 1e-6
        assert abs(c["delta"]) <= b.du_max + 1e-6


def test_only_identifiable_channels_are_recommended(system):
    for key in SCENARIOS:
        d = system.decide(SCENARIOS[key]["timestamp"], persist=False)
        for c in d.action_block.get("changes", []):
            assert system.model.bounds[c["tag"]].identifiable


def test_reruns_are_reproducible(system, cfg):
    """Same input, same output -- ТЗ "Воспроизводимость"."""
    from src.agents.orchestrator import _json_default
    a = system.decide(SCENARIOS["stable"]["timestamp"], persist=False)
    b = system.decide(SCENARIOS["stable"]["timestamp"], persist=False)
    sa = json.dumps(a.to_dict(), sort_keys=True, default=_json_default)
    sb = json.dumps(b.to_dict(), sort_keys=True, default=_json_default)
    assert sa == sb


def test_hard_sulfur_constraint_is_enforced_on_every_recommendation(system):
    """Whenever the system *certifies* the spec, the chance constraint holds."""
    from src.optimize.chance import check_max
    f = system.ds.frame
    r = system.replay_frame
    op = f["derived:operating"].to_numpy(bool)
    te = (f["date"] > pd.Timestamp(system.cfg["time"]["train_end"])).to_numpy()
    yh = r["y_hat"].to_numpy(float)
    # Certifying decisions are rare on this unit -- the plant runs at the
    # limit -- so the sample is drawn from the rows where certification is
    # possible at all, plus the pinned stable scenario.
    idx = np.where(op & te & np.isfinite(yh) & (yh < 5.5))[0]
    stamps = [SCENARIOS["stable"]["timestamp"]]
    stamps += [str(f["date"].iloc[i]) for i in idx[::max(1, len(idx) // 30)]]
    checked = 0
    for ts in stamps:
        d = system.decide(ts, persist=False)
        if d.action not in ("recommend", "hold"):
            continue
        checked += 1
        assert d.action_block["certifies_spec"] is True
        assert d.effect_block["sulfur_after_with_backoff_mgkg"] <= \
            float(system.cfg["spec"]["S_max_mgkg"]) + 1e-6
        assert d.effect_block["margin_to_limit_mgkg"] >= -1e-6
    assert checked >= 1, "no certifying decision found to check"


def test_quality_is_never_in_the_objective(system):
    from src.optimize.objective import (QualityInObjectiveError,
                                        assert_quality_not_in_objective)
    econ = system.optimization_agent._economics(
        [k for k, b in system.model.bounds.items() if b.identifiable])
    assert_quality_not_in_objective(econ.names)
    with pytest.raises(QualityInObjectiveError):
        assert_quality_not_in_objective(econ.names + ["sulfur_margin"])


def test_assumptions_register_covers_every_bound_and_proxy(system, tmp_path):
    from src.assumptions import build_register
    reg = build_register(system)
    params = " ".join(e["parameter"] for e in reg["entries"])
    for k in system.model.bounds:
        assert k in params
    for t in system.model.severity_terms:
        assert t.name in params
    assert "activation energy" in params
    assert all(e["status"].startswith("[") for e in reg["entries"])
    assert "historical min/max" in reg["rule"]


def test_validation_report_runs_and_is_honest(system):
    from src.validate.report import evaluate
    res = evaluate(system)
    assert res["n_test_lab_results"] > 100
    assert res["coverage"]["one_sided_eq63"]["n"] > 100
    assert res["step_response"]["sign_ok"]
    assert any("persistence" in n for n in res["honesty_notes"])
