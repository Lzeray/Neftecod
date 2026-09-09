"""Time splits and leakage guards -- model_spec.md §38; ТЗ §6.

Random row shuffling is forbidden.  Every split is by time, and
:func:`assert_no_leakage` fails if any test timestamp precedes any train
timestamp.  Repeated LIMS values are collapsed to one row per measurement
(§25) before any metric or fit sees them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class TemporalLeakageError(AssertionError):
    """Raised when a split would leak future information."""


@dataclass
class TimeSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self, t: pd.Series) -> dict:
        return {"n_train": int(self.train_idx.size),
                "n_test": int(self.test_idx.size),
                "train_range": [str(t.iloc[self.train_idx].min()),
                                str(t.iloc[self.train_idx].max())]
                if self.train_idx.size else [],
                "test_range": [str(t.iloc[self.test_idx].min()),
                               str(t.iloc[self.test_idx].max())]
                if self.test_idx.size else []}


def time_split(t, train_end, test_end=None, embargo_h: float = 0.0) -> TimeSplit:
    """Split by timestamp with an optional embargo gap.

    ``embargo_h`` drops a window right after ``train_end`` so that a sample
    whose response is still being driven by training-window inputs cannot end
    up in the test set.
    """
    t = pd.to_datetime(pd.Series(np.asarray(t)).reset_index(drop=True))
    tr_end = pd.Timestamp(train_end)
    te_end = pd.Timestamp(test_end) if test_end is not None else t.max()
    emb = tr_end + pd.to_timedelta(float(embargo_h), unit="h")
    train = np.where(t <= tr_end)[0]
    test = np.where((t > emb) & (t <= te_end))[0]
    return TimeSplit(train, test, tr_end, te_end)


def assert_no_leakage(t, train_idx, test_idx) -> None:
    """Fail if any test timestamp is not strictly after every train timestamp."""
    t = pd.to_datetime(pd.Series(np.asarray(t)).reset_index(drop=True))
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    t_train_max = t.iloc[np.asarray(train_idx)].max()
    t_test_min = t.iloc[np.asarray(test_idx)].min()
    if t_test_min <= t_train_max:
        raise TemporalLeakageError(
            f"temporal leakage: test starts at {t_test_min} but training data "
            f"runs to {t_train_max}")
    overlap = set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())
    if overlap:
        raise TemporalLeakageError(f"{len(overlap)} rows appear in both folds")


def rolling_origin(t, n_folds: int = 4, min_train_frac: float = 0.5,
                   embargo_h: float = 0.0) -> list[TimeSplit]:
    """Expanding-window (rolling-origin) folds -- the only valid CV here."""
    t = pd.to_datetime(pd.Series(np.asarray(t)).reset_index(drop=True))
    n = len(t)
    if n < 10 or n_folds < 1:
        raise ValueError("not enough data for rolling-origin validation")
    start = int(n * min_train_frac)
    edges = np.linspace(start, n - 1, n_folds + 1).astype(int)
    out = []
    for i in range(n_folds):
        sp = time_split(t, t.iloc[edges[i]], t.iloc[edges[i + 1]], embargo_h)
        if sp.train_idx.size and sp.test_idx.size:
            assert_no_leakage(t, sp.train_idx, sp.test_idx)
            out.append(sp)
    return out


def unique_measurements(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per underlying measurement (§25): drop the as-of repeats."""
    col = f"{name}__new"
    if col not in df.columns:
        raise ValueError(f"{col} missing; build the frame with sync.asof_join")
    return df.loc[df[col].astype(bool)].copy()
