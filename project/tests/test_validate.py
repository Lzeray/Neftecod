"""Isolation checks for Block K -- model_spec.md §38 (eq 67-69).

Targets:
  * splits.time_split / assert_no_leakage -- a test timestamp may never precede
    a training timestamp; repeated LIMS values are not independent samples
  * metrics (eq 67-69) -- hand-computed values on a tiny array; the persistence
    baseline is reported explicitly
  * coverage -- empirical coverage of a correctly specified interval is ~1-alpha
"""
import numpy as np
import pandas as pd
import pytest

from src.data import sync
from src.validate import coverage, metrics, splits


# ------------------------------------------------------- splits
def _t(n=100):
    return pd.Series(pd.date_range("2023-01-01", periods=n, freq="10min"))


def test_time_split_is_ordered_and_leak_free():
    t = _t(100)
    sp = splits.time_split(t, t.iloc[59])
    assert sp.train_idx.max() == 59 and sp.test_idx.min() == 60
    splits.assert_no_leakage(t, sp.train_idx, sp.test_idx)


def test_leakage_guard_fires_on_a_shuffled_split():
    t = _t(50)
    rng = np.random.default_rng(0)
    idx = rng.permutation(50)
    with pytest.raises(splits.TemporalLeakageError):
        splits.assert_no_leakage(t, idx[:25], idx[25:])


def test_leakage_guard_fires_on_overlapping_folds():
    t = _t(50)
    with pytest.raises(splits.TemporalLeakageError):
        splits.assert_no_leakage(t, np.arange(30), np.arange(25, 50))


def test_embargo_removes_the_boundary_window():
    t = _t(100)
    sp = splits.time_split(t, t.iloc[59], embargo_h=1.0)
    assert sp.test_idx.min() == 66          # 6 samples of 10 min embargoed
    splits.assert_no_leakage(t, sp.train_idx, sp.test_idx)


def test_rolling_origin_folds_are_expanding_and_leak_free():
    t = _t(200)
    folds = splits.rolling_origin(t, n_folds=3)
    assert len(folds) == 3
    sizes = [f.train_idx.size for f in folds]
    assert sizes == sorted(sizes)
    for f in folds:
        splits.assert_no_leakage(t, f.train_idx, f.test_idx)
    with pytest.raises(ValueError):
        splits.rolling_origin(_t(5))


def test_repeated_lims_values_are_not_independent_samples():
    base = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=20,
                                               freq="10min")})
    src = pd.DataFrame({"timestamp": pd.to_datetime(
        ["2023-01-01 00:05:00", "2023-01-01 01:05:00"]), "value": [9.0, 8.0]})
    df = sync.asof_join(base, src, "lab")
    assert df["lab"].notna().sum() > 2
    assert len(splits.unique_measurements(df, "lab")) == 2
    with pytest.raises(ValueError):
        splits.unique_measurements(base, "lab")


# ------------------------------------------------------- eq 67-69
def test_metrics_hand_computed():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 1.0, 3.5])
    # errors: -0.5, 1.0, -0.5
    assert metrics.mae(yt, yp) == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics.rmse(yt, yp) == pytest.approx(np.sqrt((0.25 + 1 + 0.25) / 3))
    assert metrics.bias(yt, yp) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        metrics.errors([1.0], [1.0, 2.0])


def test_metrics_ignore_nan_and_handle_empty():
    assert metrics.mae([1.0, np.nan], [1.0, 5.0]) == pytest.approx(0.0)
    assert np.isnan(metrics.mae([np.nan], [np.nan]))


def test_persistence_baseline_alignment():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    p = metrics.persistence(y, 2)
    assert list(p) == [1.0, 2.0]           # predicts y[2], y[3]
    with pytest.raises(ValueError):
        metrics.persistence(y, 0)
    assert metrics.persistence(y, 10).size == 0


def test_horizon_table_reports_skill_against_persistence():
    rng = np.random.default_rng(1)
    n = 500
    y = np.cumsum(rng.normal(size=n)) * 0.1 + 8.0
    # a "perfect" model at h=1 must beat persistence
    preds = {1: y[1:], 3: y[3:]}
    tab = metrics.horizon_table(y, preds)
    assert set(tab["h"]) == {1, 3}
    assert (tab["skill_MAE"] > 0.9).all()
    # an honest report when the model is *worse* than persistence
    bad = {1: np.full(n - 1, 8.0)}
    tab2 = metrics.horizon_table(y, bad)
    assert tab2["skill_MAE"].iloc[0] < 0


def test_step_response_metric_reports_gain_error():
    yt = np.array([0.0, 0.5, 0.8, 1.0])
    yp = np.array([0.0, 0.4, 0.7, 0.9])
    m = metrics.step_response_error(yt, yp)
    assert m["gain_true"] == pytest.approx(1.0)
    assert m["gain_pred"] == pytest.approx(0.9)
    assert m["gain_rel_err"] == pytest.approx(0.1)


# ------------------------------------------------------- coverage
def test_coverage_of_a_correctly_specified_interval():
    rng = np.random.default_rng(2)
    n = 20000
    yp = np.zeros(n)
    sigma = 1.3
    yt = yp + rng.normal(0, sigma, n)
    out = coverage.interval_coverage(yt, yp, sigma, alpha=0.05)
    assert out["coverage"] == pytest.approx(0.95, abs=0.01)
    assert out["honest"]


def test_coverage_detects_an_overconfident_sigma():
    rng = np.random.default_rng(3)
    n = 20000
    yt = rng.normal(0, 3.0, n)
    out = coverage.interval_coverage(yt, np.zeros(n), 0.5, alpha=0.05)
    assert out["coverage"] < 0.5 and not out["honest"]


def test_one_sided_coverage_matches_the_chance_constraint_level():
    rng = np.random.default_rng(4)
    n = 40000
    yt = rng.normal(0, 1.0, n)
    out = coverage.one_sided_coverage(yt, np.zeros(n), 1.0, alpha=0.05)
    assert out["coverage"] == pytest.approx(0.95, abs=0.01)


def test_violation_rates_hand_example():
    yt = np.array([8.0, 11.0, 9.0, 12.0])
    yp = np.array([8.0, 8.0, 9.9, 11.0])
    out = coverage.violation_rates(yt, yp, limit=10.0, backoff=0.5)
    # flagged: yp+0.5 > 10 -> [F, F, T, T]; actual: [F, T, F, T]
    assert out["TP"] == 1 and out["FP"] == 1 and out["FN"] == 1 and out["TN"] == 1
    assert out["missed_rate"] == pytest.approx(0.5)
    assert out["actual_violations"] == 2


def test_coverage_empty_input():
    assert coverage.interval_coverage([], [], 1.0)["n"] == 0
    assert coverage.violation_rates([], [], 10.0)["n"] == 0
