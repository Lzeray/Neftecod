"""Isolation checks for Block A -- model_spec.md §1, §23-25 (eq 1-2, 53-54).

Targets:
  * loaders.detect_sentinels / load_telemetry -- DCS bad-value masking
  * loaders._parse_paired_sheet via load_lims/load_pak -- wide -> tidy long
  * sync.asof_join                            -- eq 54 (backward merge + age)
  * staleness.rolling_stale                   -- §23 stuck-sensor detector
"""
import numpy as np
import pandas as pd
import pytest

from src.data import loaders, staleness, sync


# ---------------------------------------------------------------- loaders
def test_detect_sentinels_finds_shared_code_and_spares_zero():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "A": rng.normal(100, 5, n),
        "B": rng.normal(3, 0.2, n),
        "C": rng.normal(50, 1, n),
    })
    df.loc[:299, "A"] = 307.0        # shared bad-value code, 2 columns
    df.loc[:299, "B"] = 307.0
    df.loc[300:599, "C"] = 0.0       # genuine shutdown, must survive
    found = loaders.detect_sentinels(df, min_repeats=200, robust_z=6.0,
                                     min_columns=2)
    assert found == [307.0]
    assert 0.0 not in found


def test_detect_sentinels_ignores_single_column_plateau():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "A": rng.normal(100, 5, n),
        "B": rng.normal(3, 0.2, n),
    })
    df.loc[:299, "A"] = 999.0        # only one column -> not a plant-wide code
    assert loaders.detect_sentinels(df, 200, 6.0, min_columns=2) == []


def test_load_telemetry_drops_index_columns_and_masks(tmp_path):
    n = 600
    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "Unnamed: 0.1": range(n),
        "date": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "T1": np.linspace(100, 110, n),
        "F2": np.linspace(10, 12, n),
    })
    df.loc[:249, "T1"] = 307.0
    df.loc[:249, "F2"] = 307.0
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    out, rep = loaders.load_telemetry(p, cleaning_cfg={"sentinel_min_repeats": 200})
    assert not [c for c in out.columns if c.startswith("Unnamed")]
    assert rep["dropped_index_columns"] == ["Unnamed: 0", "Unnamed: 0.1"]
    assert 307.0 in rep["sentinels"]
    assert out["T1"].isna().sum() == 250
    assert out["date"].is_monotonic_increasing


def test_load_telemetry_edge_cases(tmp_path):
    p = tmp_path / "e.csv"
    pd.DataFrame({"date": ["2023-01-01 00:00:00"], "T1": [1.0]}).to_csv(p, index=False)
    out, rep = loaders.load_telemetry(p)
    assert len(out) == 1 and rep["sentinels"] == []
    # duplicate timestamps collapse, last wins
    pd.DataFrame({"date": ["2023-01-01 00:00:00", "2023-01-01 00:00:00"],
                  "T1": [1.0, 2.0]}).to_csv(p, index=False)
    out, _ = loaders.load_telemetry(p)
    assert len(out) == 1 and out["T1"].iloc[0] == 2.0


def test_paired_sheet_parser_handles_blank_separator(tmp_path):
    # layout: (ts,val) | blank | (ts,val)   -- the ПАК workbook shape
    raw = pd.DataFrame(index=range(5), columns=range(5), dtype=object)
    raw.iloc[0, 0] = "S1"; raw.iloc[1, 0] = "ppm"
    raw.iloc[0, 3] = "S2"; raw.iloc[1, 3] = "кг/м3"
    for i, r in enumerate(range(2, 5)):
        raw.iloc[r, 0] = f"2023-01-01 0{i}:00:00"; raw.iloc[r, 1] = 1.0 + i
        raw.iloc[r, 3] = f"2023-01-02 0{i}:00:00"; raw.iloc[r, 4] = 800.0 + i
    p = tmp_path / "pak.xlsx"
    raw.to_excel(p, header=False, index=False)
    ls = loaders.load_pak(p)
    assert set(ls.frame["param"]) == {"S1", "S2"}
    assert len(ls.frame) == 6


def test_sulfur_unit_normalisation():
    assert loaders.to_mgkg(np.array([8.0]), "ppm")[0] == pytest.approx(8.0)
    assert loaders.to_mgkg(np.array([0.2]), "% масс.")[0] == pytest.approx(2000.0)
    with pytest.raises(ValueError):
        loaders.to_mgkg(np.array([1.0]), "°С")


# ---------------------------------------------------------------- sync
def _base(n=10, start="2023-01-01 00:00:00"):
    return pd.DataFrame({"date": pd.date_range(start, periods=n, freq="10min")})


def test_asof_join_known_answer_and_age():
    base = _base(6)                       # 00:00 .. 00:50
    src = pd.DataFrame({"timestamp": pd.to_datetime(
        ["2023-01-01 00:05:00", "2023-01-01 00:35:00"]), "value": [1.0, 2.0]})
    out = sync.asof_join(base, src, "lab")
    # hand answer: NaN, 1, 1, 1, 2, 2
    assert np.isnan(out["lab"].iloc[0])
    assert list(out["lab"].iloc[1:]) == [1.0, 1.0, 1.0, 2.0, 2.0]
    # ages in hours: 5, 15, 25 min then 5, 15 min
    assert out["lab__age_h"].iloc[1] == pytest.approx(5 / 60)
    assert out["lab__age_h"].iloc[3] == pytest.approx(25 / 60)
    assert out["lab__age_h"].iloc[4] == pytest.approx(5 / 60)


def test_asof_join_never_leaks_future():
    base = _base(6)
    src = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:55:00"]),
                        "value": [42.0]})
    out = sync.asof_join(base, src, "lab")
    assert out["lab"].isna().all(), "a future measurement must never appear"


def test_asof_join_row_index_misalignment_cannot_leak():
    """Deliberately shuffle the source rows: the answer must not change."""
    base = _base(6)
    src = pd.DataFrame({"timestamp": pd.to_datetime(
        ["2023-01-01 00:05:00", "2023-01-01 00:35:00"]), "value": [1.0, 2.0]})
    a = sync.asof_join(base, src, "lab")
    b = sync.asof_join(base, src.iloc[::-1].reset_index(drop=True), "lab")
    pd.testing.assert_frame_equal(a, b)


def test_asof_join_repeated_value_counted_once():
    base = _base(6)
    src = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:05:00"]),
                        "value": [7.0]})
    out = sync.asof_join(base, src, "lab")
    assert out["lab"].notna().sum() == 5          # repeated across rows
    assert out["lab__new"].sum() == 1             # but one measurement only
    assert len(sync.dedup_new_only(out, "lab")) == 1


def test_asof_join_turnaround_delays_availability():
    base = _base(6)
    src = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:05:00"]),
                        "value": [7.0]})
    out = sync.asof_join(base, src, "lab", turnaround_h=0.5)
    # available only from 00:35 on -> rows at 00:40, 00:50
    assert out["lab"].notna().sum() == 2
    assert out["lab__age_h"].iloc[4] == pytest.approx(35 / 60)


def test_asof_join_max_age_masks_and_empty_source():
    base = _base(6)
    src = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:05:00"]),
                        "value": [7.0]})
    out = sync.asof_join(base, src, "lab", max_age_h=20 / 60)
    assert out["lab"].notna().sum() == 2          # 00:10 and 00:20 only
    empty = pd.DataFrame({"timestamp": pd.to_datetime([]), "value": []})
    out2 = sync.asof_join(base, empty, "lab")
    assert out2["lab"].isna().all() and not out2["lab__new"].any()


def test_align_telemetry_namespaces_collide_free():
    a = _base(3); a["T6"] = [1.0, 2.0, 3.0]
    h = _base(3); h["T6"] = [10.0, 20.0, 30.0]
    out = sync.align_telemetry(a, h)
    assert "AVT:T6" in out.columns and "HDS:T6" in out.columns
    assert list(out["AVT:T6"]) == [1.0, 2.0, 3.0]
    assert list(out["HDS:T6"]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------- staleness
def test_rolling_stale_flags_constant_segment_only():
    s = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 50),
                        np.full(30, 5.0)])
    st = staleness.rolling_stale(s, window=10)
    assert not st.iloc[:50].any()
    assert st.iloc[-1] and st.iloc[60:].all()


def test_rolling_stale_partial_window_is_not_stale():
    s = pd.Series(np.full(5, 3.0))
    assert not staleness.rolling_stale(s, window=10).any()


def test_flatline_run_length():
    s = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0])
    assert list(staleness.flatline_run_length(s)) == [1, 2, 3, 1, 2]


def test_robust_z_and_outlier_mask_resist_outliers():
    s = pd.Series(np.r_[np.zeros(99), [1000.0]])
    z = staleness.robust_z(s)
    assert abs(z.iloc[0]) < 1e-9
    assert staleness.outlier_mask(s).sum() == 1
    # degenerate MAD must not divide by zero
    assert (staleness.robust_z(pd.Series(np.ones(10))) == 0).all()
