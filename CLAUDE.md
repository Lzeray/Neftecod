# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A chance-constrained multi-agent decision system for the diesel chain
**ЭЛОУ-АВТ-6 → 24-2000 (гидроочистка) → блендинг**. It implements
`prompt/model_spec.md` (Parts I–IX, equations 1–70) against a real plant data
package. The hard requirement is product diesel sulfur ≤ 10 mg/kg.

**`prompt/model_spec.md` is the single source of truth for the mathematics.**
Where code and spec disagree, the spec wins — stop and flag the conflict rather
than guessing. `prompt/coding_agent_prompt.md` states the working rule: every
block is verified in isolation *before* integration, and no block ships without
its pytest module.

Almost every module docstring cites the equations it implements. When changing a
module, keep those citations accurate — they are how the code stays checkable
against the spec.

## Layout

```
ха/                      # repo root (not a git repo)
  prompt/                # model_spec.md + coding_agent_prompt.md — the spec
  data/                  # avt_tags.csv, 242000_tags.csv (189k rows each)
  *.xlsx, *.pdf, *.docx  # LIMS, ПАК, tag dictionary, schematics, ТЗ
  project/               # all code; run every command from here
```

## Commands

Run from `project/`.

```bash
python -m pytest                                # 237 tests, ~60 s
python -m pytest tests/test_ident.py            # one module
python -m pytest tests/test_ident.py::test_ols_is_biased_and_iv_recovers_the_true_gain_closed_loop
python -m pytest -k "leakage or coverage"       # by name

python -m src.cli report                        # identify + validate + assumptions + verification + demo
python -m src.cli identify                      # §37 identification protocol
python -m src.cli validate                      # §38 held-out validation
python -m src.cli decide --at "2026-07-12 14:00:00"
python -m src.cli demo                          # the three pinned scenarios
python -m src.cli assumptions                   # regenerate artifacts/assumptions.yaml
python -m src.cli verification                  # regenerate artifacts/verification_report.md
python -m src.dashboard --at "2026-07-12 14:00:00"

python -m src.cli <cmd> --no-cache              # rebuild from scratch (~90 s)
```

`tests/test_system.py` is the only module that touches the real data package; it
skips itself if the files are absent, so the rest of the suite runs anywhere.

Caches live in `artifacts/cache/` keyed by a hash of inputs + config. **Delete
them (or pass `--no-cache`) after changing anything in `src/data/`,
`src/ident/`, or `config/config.yaml`** — a stale `identified_model.pkl` will
silently mask your change.

## Architecture

Data flows in one direction; each stage is independently testable.

```
build_dataset()      src/data/dataset.py    → one time-indexed frame, cached parquet
   ↓
identify()           src/ident/pipeline.py  → IdentifiedModel (kernel, gains, bounds,
                                              severity terms, filter params), cached pickle
   ↓
build_system()       src/system.py          → PlantPredictor + causal replay + the 3 agents
   ↓
Orchestrator.decide()                       → Decision (recommend | hold | mitigate | refuse)
```

`src/system.py` is the **only** place that wires blocks together. Everything
else must stay usable standalone.

### The predictor (eq 55)

`ŷ = g_φ(kernel) + r_ψ(residual) + η(offset anchored on LIMS)`.

- `g_φ` — `src/kernels/sulfur_kernel.py`, the reduced plug-flow form of eq 16.
- `r_ψ` — sign-constrained robust regression on lagged/shaped inputs.
- `η` — scalar bias filter (eq 51–52), re-anchored at each accepted lab result.

**Both `g_φ` and `r_ψ` read shaped inputs, never instantaneous ones** (§26). The
kernel's four inputs arrive as `kin:*` columns in the `utilde` frame, each
routed through the FOPTD of the channel that drives it
(`channels.KERNEL_INPUT_DRIVERS`). Identified delays are 2.0 h (reactor
temperature), 4.5 h (feed rate), 5.8 h (loop pressure) — which is why
`horizon.H` is 42 samples (7 h) and why every recommendation carries a
`response_timing` block. If you add a kernel input, add its driver too.

`src/predict/plant.py::replay()` runs the whole thing causally over the record:
it iterates only over anchor events and fills η/P between them vectorised. The
value stored on a lab row is the one available **before** that result was used,
so lab rows carry genuine out-of-sample forecasts.

### Agents (ТЗ §3)

The three agents communicate **only** through the dataclasses in
`src/agents/contracts.py`. `tests/test_agents.py` runs the orchestrator with
every agent replaced by a stub — keep that possible.

The orchestrator gate is eq 66: admissible set non-empty ∧ data fresh ∧
σ_pred ≤ σ_max. There are four outcomes, and `mitigate` is the important one:
this plant cannot certify 10 mg/kg at 95% confidence almost anywhere — 17% of
held-out lab results exceed the limit, and only 4 of 53 640 held-out operating
samples admit a certifying recommendation (95% `mitigate` / 5% `refuse` on a
uniform scan). `𝒞` is therefore usually empty and §34's lexicographic stage 1
returns the best risk reduction labelled `certifies_spec: false`. Never let a
`mitigate` path present itself as meeting the specification, and do not treat
"the demo mostly mitigates" as a bug to be tuned away.

## Invariants that are easy to break

These each cost real debugging time and are each pinned by a test.

- **Time-only synchronisation.** `sync.asof_join` merges backwards on timestamp
  and attaches `__age_h` and `__new`. Row-index alignment is forbidden.
  Repeated LIMS values are not independent samples — every fit and metric goes
  through `__new` / `splits.unique_measurements`.
- **Quality never enters the objective.** `objective.assert_quality_not_in_objective`
  raises on any spec-shaped name. It is called from `EconomicModel.__post_init__`.
- **Model comparisons must be paired and carry a standard error.** The MAE
  standard error here is ~0.28 mg/kg on 468 held-out results, larger than most
  differences between candidate models — `metrics.paired_mae_difference` exists
  so a report cannot claim skill it cannot demonstrate. The current honest
  verdict: indistinguishable from persistence, significantly worse than a
  constant.
- **Residual sign constraints.** `PHYSICS_SIGNS` in `src/ident/pipeline.py`. The
  residual may correct the *magnitude* of a kinetic effect, never its direction.
  On this data all five HDS coefficients came out with the impossible sign and
  were clipped to zero. Clipping also re-anchors the intercept — don't remove that.
- **`Ea` is fixed at a literature value**, not fitted. A free fit drives it to
  its lower bound and destroys the kinetic sign the whole recommendation rests on.
- **Delay-line length.** `features/lag.foptd` produces the strictly causal ZOH
  numerator `[0, K(1−a)]`, so `statespace.build_state_space` needs `n+1` delay
  stages, not `n`. Identification and prediction must agree sample-for-sample.
- **σ is calibrated by replaying the *deployed* predictor** over the training
  window — the finished kernel and residual with η running. Calibrating on the
  un-anchored model, or on a predictor assembled from mismatched parts (an
  inner-fit kernel with a full-window intercept), reports an error the system
  does not make; one such mismatch produced a 98 mg/kg constant residual and
  collapsed σ to zero.
- **Leverage is measured in the kernel's input space** (feed sulfur, T_WABT,
  p_H2, LHSV) and its variance contribution is capped (`uncertainty.lev_cap`).
  The raw Mahalanobis distance is surfaced separately as an extrapolation flag.
- **The operating box is widened to contain the current point.** The declared
  range is a conservative model range; the plant sometimes sits outside it, and
  demanding an instantaneous jump back would make every such cycle infeasible.
- **Hyperparameters are selected on an inner time split** of the training
  window, never on the rows the residual was fitted on.
- **ПАК usability comes from identification, not config.** The filter search
  rejected it, so it may not satisfy the eq 66 freshness clause either.

## Data quirks this package actually has

Discovered and handled; don't "fix" them back.

- **DCS sentinels** `213, 240, 251, 252, 307, 313` plus a 0–10 dP rail, detected
  data-driven. `0.0` is a real shutdown reading and is never a sentinel.
- **`AVT:D10` (crude density)** has 10 non-null rows, all `240.0`. Crude density,
  sulfur and water are unmeasured and live only in the robust set `𝒰` (eq 64).
- **Five `КИП` descriptions contradict their data** (e.g. `HDS:T6` documented as
  a sulfur analyser reads ~363 °C). The observed physical kind wins;
  `src/data/tags.py` records the conflict.
- **The LIMS unit row is shifted** relative to the parameter row. Units come from
  the parameter name.
- **Feed sulfur is `Mass.Sulfur` in % mass** at the HDS feed point (~130 results),
  a different parameter name and unit from the product's `Mg.Sulfur` in mg/kg.
- **ВАК formulas are not machine-readable as written**: Russian decimal commas,
  `x` as multiplication, one unbalanced bracket, one truncated expression. Only
  2 of 8 beat a training-median baseline on held-out LIMS
  (`src/validate/vak_audit.py`); the rest are disabled, not patched.
- **`corr(T_WABT, S_product) = +0.184`** — the sign kinetics forbid. This is the
  closed-loop bias of §18, and the package contains no valid instrument for it.

## Where to change what

- `config/config.yaml` — paths, time split, α, σ_max, horizons, weights, rate
  limits, blend definition. Nothing hard-codes a data path.
- `src/domain.py` — which tags are u / d / y, with the provenance of each choice
  (schematics + `КИП` dictionary). Start here to add or retire a channel.
- `src/ident/pipeline.py` — the §37 protocol, in order; also `PHYSICS_SIGNS`,
  the bootstrap significance test and the bounds construction.
- `artifacts/` — every run persists inputs, agent outputs, decision traces,
  the assumptions register and the verification report. `assumptions.yaml` and
  `verification_report.md` are **generated**; edit their generators
  (`src/assumptions.py`, `src/verification_report.py`), not the output.

## Honesty requirements

The ТЗ and the spec both make this explicit, and the validation report enforces
it. `artifacts/validation_report.json` carries a `honesty_notes` field that
states plainly what the model does *not* do — currently that it beats the
mandated persistence baseline (eq 69) but not a constant at the training median.
Any new claim about performance belongs there with its held-out number, and any
coefficient the package does not confirm belongs in `assumptions.yaml` with a
`[БАЛАНС] / [ЛИТ] / [ВЫВОД] / [ДОП]` status. A historical min/max is never
presented as an industrial limit.
