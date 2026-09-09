"""The three reproducible demonstration scenarios -- ТЗ §6; prompt Block L.

  1. a stable period where the system proposes little or no action;
  2. a period at the specification limit where it acts;
  3. a period with stale data where it refuses, with an explanation.

The timestamps are pinned so a re-run reproduces byte-identical artifacts.  All
three lie inside the **test** window (after ``time.train_end``), so none of them
was used to fit anything.
"""
from __future__ import annotations

SCENARIOS = {
    "stable": {
        "timestamp": "2026-07-12 14:00:00",
        "title": "Stable period, margin to the specification",
        "expect": ("A low sulfur forecast leaves room under the 10 mg/kg "
                   "chance constraint, so the admissible set is non-empty and "
                   "the optimiser only fine-tunes the regime, with every hard "
                   "constraint satisfied. This state is RARE: it holds in 4 of "
                   "53 640 held-out operating samples with a fresh laboratory "
                   "result. The unit spends almost all of its time unable to "
                   "certify the limit at 95% confidence."),
    },
    "at_limit": {
        "timestamp": "2025-07-01 00:00:00",
        "title": "Operating at the specification limit",
        "expect": ("The forecast plus the statistical back-off of eq 63 "
                   "exceeds 10 mg/kg: the admissible set is empty, so the "
                   "system falls back to the lexicographic risk-reduction "
                   "stage and states plainly that the limit is not certified."),
    },
    "stale_data": {
        "timestamp": "2026-04-27 09:00:00",
        "title": "Stale laboratory data",
        "expect": ("The last laboratory sulfur result is over 300 h old and "
                   "the stream analyser cannot stand in, so the freshness "
                   "clause of eq 66 fails and the system refuses."),
    },
}
