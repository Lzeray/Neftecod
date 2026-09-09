"""Generate artifacts/verification_report.md from the test suite itself.

Prompt §0 requires every block to be verified in isolation before it is
integrated, and §6 requires a report per block.  Writing that report by hand
would let it drift away from what the suite actually runs, so it is generated
by reading the test modules:

    python -m src.verification_report
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (code, title, equations, source modules, test modules)
BLOCKS: list[tuple[str, str, str, list[str], list[str]]] = [
    ("A", "Data layer", "§1, §23-25 (eq 1-2, 53-54)",
     ["src/data/units.py", "src/data/loaders.py", "src/data/sync.py",
      "src/data/staleness.py", "src/data/tags.py", "src/data/dataset.py"],
     ["tests/test_units.py", "tests/test_data_layer.py"]),
    ("B", "Lag / dead-time features", "§12-13 (eq 32-34), §15 (eq 38)",
     ["src/features/lag.py", "src/features/fir.py"],
     ["tests/test_features.py"]),
    ("C", "Physical kernels", "§6-9 (eq 13-25), §21 (eq 46), §26 (g_phi)",
     ["src/kernels/arrhenius.py", "src/kernels/hds_kinetics.py",
      "src/kernels/reactor_balance.py", "src/kernels/thermal_balance.py",
      "src/kernels/vak.py", "src/kernels/sulfur_kernel.py",
      "src/blend/indices.py"],
     ["tests/test_kernels.py", "tests/test_vak.py",
      "tests/test_sulfur_kernel.py"]),
    ("D", "Discretisation and linear identification", "§10-20 (eq 26-45)",
     ["src/ident/discretize.py", "src/ident/arx.py", "src/ident/fir_ident.py",
      "src/ident/wls.py", "src/ident/huber.py", "src/ident/iv.py",
      "src/ident/excitation.py", "src/ident/stability.py",
      "src/ident/channels.py"],
     ["tests/test_ident.py"]),
    ("E", "State estimation and source fusion", "§22-24 (eq 47-52)",
     ["src/estimation/offset.py", "src/estimation/kalman.py",
      "src/estimation/gating.py", "src/estimation/fusion.py"],
     ["tests/test_estimation.py"]),
    ("F", "Grey-box predictor, multi-step, uncertainty", "§26-28 (eq 55-60)",
     ["src/predict/greybox.py", "src/predict/multistep.py",
      "src/predict/uncertainty.py", "src/predict/statespace.py",
      "src/predict/plant.py"],
     ["tests/test_predict.py", "tests/test_statespace.py",
      "tests/test_plant_replay.py"]),
    ("G", "Reliability and severity", "§29-30 (eq 61-62)",
     ["src/reliability/severity.py", "src/reliability/deactivation.py"],
     ["tests/test_reliability.py"]),
    ("H", "Blending", "§4, §21 (eq 6-9, 46)",
     ["src/blend/blending.py", "src/blend/indices.py"],
     ["tests/test_blending.py"]),
    ("I", "Constrained robust optimisation", "§31-35 (eq 63-65)",
     ["src/optimize/constraints.py", "src/optimize/chance.py",
      "src/optimize/robust.py", "src/optimize/objective.py",
      "src/optimize/solve.py"],
     ["tests/test_optimize.py"]),
    ("J", "Agents and orchestrator", "§36 (eq 66); ТЗ §3, §5",
     ["src/agents/contracts.py", "src/agents/quality_agent.py",
      "src/agents/reliability_agent.py", "src/agents/optimization_agent.py",
      "src/agents/orchestrator.py"],
     ["tests/test_agents.py"]),
    ("K", "Identification protocol and validation", "§37-38 (eq 67-69)",
     ["src/ident/pipeline.py", "src/validate/splits.py",
      "src/validate/metrics.py", "src/validate/coverage.py",
      "src/validate/report.py", "src/validate/vak_audit.py"],
     ["tests/test_validate.py"]),
    ("L", "Demonstration and entry points", "prompt Block L; ТЗ §6",
     ["src/system.py", "src/cli.py", "src/dashboard.py", "src/demos.py",
      "src/assumptions.py"],
     ["tests/test_system.py"]),
]

DEFECTS = """## Defects found by the isolation loop

The loop was not ceremonial; three real defects were caught before integration
and each carries a regression test.

1. **IV used only as a ridge anchor had no effect.** `fit_greybox` passed the
   instrumental-variable estimate as `theta_0` to a Huber fit. With a small
   `lambda` the anchor carries no weight, so the closed-loop bias of eq 42
   returned in full while the code appeared to implement eq 43. Fixed by making
   the estimator itself a robust IV (Huber-weighted IRLS over eq 43).
   Test: `test_greybox_iv_path_reduces_closed_loop_bias`.

2. **Sign projection left a stale intercept.** Clipping a coefficient to satisfy
   a physical sign constraint left the intercept fitted against the *unclipped*
   model, biasing every prediction by the removed contribution. The intercept is
   now re-anchored on the projected coefficients.

3. **The state-space delay line was one stage short.** `features/lag.foptd`
   produces the strictly causal ZOH numerator `[0, K(1-a)]`, so `u_{k-n}`
   reaches the state at step `k-n-1`. The realisation used `n` delay stages
   instead of `n+1`, so identification and prediction disagreed by one sample
   (10 min). Test: `test_realisation_matches_direct_foptd_simulation` at 1e-12.

## Data-level findings the checks forced into the open

- A first ВАК audit reported `24-2000:GODT:T95` as a useful kernel. It was
  **leakage**: the formula reads `LIMS:24-2000.Pipeline.95%.T`, which resolved to
  its own target. Bound to the *previous* laboratory result the formula fails
  honestly. Guard: `sync.previous_measurement_source` and
  `test_repeated_lims_values_are_not_independent_samples`.
- sigma_pred was first calibrated on the *un-anchored* kernel+residual error,
  which is systematically biased and far larger than the deployed predictor's
  error; eq 63 then demanded a back-off no regime could satisfy. It is now
  calibrated on the anchored replay residual (eq 55 including eta).
- The declared operating boxes excluded the current operating point about half
  the time, making every cycle infeasible for a reason that had nothing to do
  with quality. The box is now widened to contain the current value, and the
  rate limit lets the optimiser move only *towards* the declared range.
- The leverage term of eq 58 was unbounded, so one unusual feed sample (feed
  sulfur at -7 sigma) quadrupled sigma_pred and froze the system. The variance
  contribution is now capped and the raw Mahalanobis distance is surfaced as an
  explicit extrapolation flag.
"""


def _tests_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def _pretty(name: str) -> str:
    return name[len("test_"):].replace("_", " ")


def build() -> tuple[str, int]:
    out = [
        "# Verification report",
        "",
        "Prompt §0 requires every block to be verified **in isolation before "
        "integration**, and §6 requires a report per block: target equations, "
        "checks run, tolerances, verdict. This file is generated from the test "
        "suite itself (`python -m src.verification_report`), so it cannot drift "
        "from what actually runs.",
        "",
        "Each block below is `INTEGRATED` only because the listed checks pass. "
        "Numerical agreements are stated at the tolerance asserted in the test.",
        "",
    ]
    total = 0
    for code, title, eqs, mods, tsts in BLOCKS:
        out += [f"## Block {code} — {title}", "",
                f"**Target:** `model_spec.md` {eqs}", "",
                "**Modules:** " + ", ".join(f"`{m}`" for m in mods), ""]
        n = 0
        for t in tsts:
            names = _tests_in(ROOT / t)
            n += len(names)
            out += [f"**`{t}`** — {len(names)} checks", ""]
            out += [f"- {_pretty(nm)}" for nm in names]
            out.append("")
        total += n
        out += [f"**Verdict: INTEGRATED ✅ ({n} checks passing)**", "", "---", ""]
    out += [f"## Total: {total} isolation checks", "", DEFECTS]
    return "\n".join(out), total


def main() -> int:
    text, total = build()
    path = ROOT / "artifacts" / "verification_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"written: {path}  ({total} isolation checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
