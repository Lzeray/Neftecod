# Coding-Agent Prompt — Implement the AVT → HDS → Blending Diesel Optimization Model

You are a senior process-control / ML engineer. Implement, in Python, the complete mathematical model described in the accompanying file **`model_spec.md`** (Parts I–IX, equations 1–70). That file is the **single source of truth**. This prompt tells you *how* to build and, above all, *how to verify* it. Where this prompt and `model_spec.md` disagree on math, `model_spec.md` wins; stop and flag the conflict instead of guessing.

---

## 0. NON-NEGOTIABLE WORKING RULE — VERIFY EVERY BLOCK IN ISOLATION

**Do not trust this prompt's generality. Do not assume any equation "obviously" works. Every code block must earn its place.**

For **each** unit of work below (each function, each equation implementation, each module), follow this loop and do not skip it:

1. **Restate the target.** Quote the exact equation number(s) and section from `model_spec.md` the block implements. Write, in one or two sentences, what the block must do and what its inputs/outputs/units are.
2. **Implement the block in isolation** in a scratch cell/file — not yet wired into the pipeline.
3. **Verify the block on its own**, independently of the rest of the system, using at minimum:
   - a **dimensional/units check** (inputs and outputs carry the units stated in `model_spec.md §1`);
   - a **closed-form / analytic check** on a tiny hand-computable example where the correct answer is known by hand;
   - **property/invariant checks** (e.g. mass fractions sum to 1; a discretized stable system has `ρ(A)<1`; a monotonic kernel is monotonic; a filter with no measurement reduces to pure prediction; sulfur blending is linear in mass);
   - **edge cases** (empty input, NaN/missing, a stuck/constant sensor, a single data point, extreme but physical values);
   - a **numerical sanity check** against an independent reference implementation where one exists (e.g. `scipy.signal` for discretization, `numpy.linalg.eig` for spectral radius, `statsmodels`/`scipy` for regression), and confirm agreement to a stated tolerance.
4. **Only if all checks pass**, integrate the block into the pipeline. Record the checks as automated tests (see §3) so they run again later.
5. If a check fails, fix the block and **re-run all of its checks from scratch** before integrating. Never integrate a block whose verification you have not just re-run.

At the end of every block, print a short **verification report**: target (eq refs), checks run, results, tolerance, and an explicit "INTEGRATED ✅ / BLOCKED ❌" verdict. Do not proceed to the next block while the current one is BLOCKED.

Treat this loop as more important than speed. A correct, individually verified block added late is worth more than a fast block that silently violates the spec.

---

## 1. Inputs (data & documents)

Files are provided in a `data/` directory; read paths from config, never hard-code:

| File | Content | Notes |
|---|---|---|
| `avt_tags.csv` | AVT telemetry, 10-min grid, ~189k rows | time key = `date`; `Unnamed:*` are junk indices — drop them |
| `242000_tags.csv` | Hydrotreater/stabilizer (unit 24-2000) telemetry, 10-min | same time grid |
| `ЛИМСы…xlsx` | LIMS lab results per sampling point/product | wide layout: per group, alternating (timestamp, value) columns; rows 0–2 are param/unit/count headers; data starts below. Rare, asynchronous, has outliers |
| `Выгрузка ПАК…xlsx` | Stream analyzers: `24-2000:Mg.Sulfur` (10-min), `24-2000:D15` | headered similarly; different coverage per signal |
| `Теги…xlsx` | Tag dictionary — sheets `КИП` (instruments), `ЛА` (lab points), `ПАК` (stream), `ВАК` (virtual analyzers) | **use this to resolve every tag's physical meaning**; do not infer meaning from the letter in a short tag name |
| AVT schematics (PDF) | P&ID-style schemes marking the manipulated variables | use to confirm which tags are actuators vs measurements |
| `model_spec.md` | The mathematical model | source of truth |

The **hard product-quality constraint** is product diesel sulfur `≤ 10 mg/kg` (LIMS point "Гидроочистка, т.2"; stream analog `24-2000:Mg.Sulfur`). The process historically runs close to this limit — the model must keep a data-quality-scaled margin.

**Load the `ВАК` sheet regression formulas** (for D15, T50, T90, T95, I250/I350, CFPP, CloudPoint, IBP/EBP, etc.) and use them as the physical/empirical **kernel** `g_φ` for fractional properties (spec §26). Parse them into evaluable expressions keyed by the `КИП` tag symbols they reference.

---

## 2. Tech stack & project layout

- Python ≥ 3.11. Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`; optimization: `cvxpy` (convex/QP path) and `casadi` or `scipy.optimize`/`ipopt` (NLP path); config: `pydantic`/`yaml`; tests: `pytest`; optional dashboard: `streamlit` or `plotly`.
- Deterministic: set and log all random seeds; pin versions in `requirements.txt`/`pyproject.toml`.
- Suggested layout (adapt if you justify it):

```
project/
  README.md                # reproducible run instructions (required by spec Part VIII)
  requirements.txt
  config/config.yaml        # paths, horizons, α, bounds, weights, seeds
  src/
    data/         loaders.py, sync.py, staleness.py, units.py
    features/     lag.py, fir.py
    kernels/      hds_kinetics.py, arrhenius.py, reactor_balance.py, thermal_balance.py, distillation.py, vak.py
    ident/        arx.py, fir_ident.py, wls.py, huber.py, iv.py, excitation.py, stability.py, discretize.py
    estimation/   offset.py, kalman.py, gating.py, fusion.py
    predict/      greybox.py, multistep.py, uncertainty.py
    reliability/  severity.py, deactivation.py
    blend/        blending.py, indices.py
    optimize/     constraints.py, chance.py, robust.py, objective.py, solve.py
    agents/       quality_agent.py, reliability_agent.py, optimization_agent.py, orchestrator.py
    validate/     metrics.py, coverage.py, splits.py
    cli.py, dashboard.py
  tests/                    # one test module per src module (see §0 and §3)
  artifacts/                # saved states, agent outputs, recommendations (auditable)
```

---

## 3. Global engineering rules (enforced by tests)

- **Time-only synchronization (spec §25, eq 54).** Build the state by *as-of / backward* merge on `date`; never align LIMS/PAK to telemetry by row index. Unit-test that a deliberately misaligned row index cannot leak a future value.
- **No temporal leakage (spec §38).** All train/test splits are by time. Provide a `time_split` util and a test that fails if any test-set timestamp precedes any train-set timestamp used to fit a parameter, and that repeated LIMS values are never counted as independent samples.
- **Explicit units (spec §1, eq 1–2).** Every physical quantity converts through documented, tested functions (°C↔K, ppm↔mass fraction, `ṁ=ρq`, `C_S` from eq 2). No implicit unit assumptions. A test checks each conversion round-trips.
- **Measurement age is a first-class feature (eq 54):** store `τ_src(t)` for LIMS/PAK; it feeds uncertainty (eq 58) and the refusal rule (eq 66).
- **Assumptions register (spec Appendix A).** Any coefficient not confirmed by the data package (kinetic params, delays, bounds, blend indices, `𝒰`, `Q,R`, `α`) is written to `artifacts/assumptions.yaml` with status `[ДОП]/[ЛИТ]/[БАЛАНС]/[ВЫВОД]` and how it was obtained. Never present a historical min/max as an industrial limit.
- **Auditability (spec Part VIII / ТЗ).** Every recommendation persists inputs, each agent's output, constraint checks, and the explanation to `artifacts/`.
- Each `src/<module>.py` has a matching `tests/test_<module>.py` containing the isolation checks from §0. CI target: `pytest` green before any integration commit.

---

## 4. Implementation plan (build in this order; verify each block per §0)

For every item, the deliverable is: implementation **+** its isolation verification report **+** its pytest module. Equation numbers refer to `model_spec.md`.

### Block A — Data layer (spec §1, §23–25)
- `loaders.py`: read telemetry (drop `Unnamed:*`, parse `date`), parse the wide LIMS/PAK sheets into tidy long frames `(timestamp, tag, value, unit)`, parse the tag dictionary and `ВАК` formulas.
- `units.py`: eq 1–2 conversions.
- `sync.py`: as-of merge, produce the aligned state frame + `τ_src` columns (eq 54).
- `staleness.py`: `stale` detector — zero/near-zero variance over a window; flag per source.
- **Isolation checks:** synthetic frames with known as-of answers; a constant PAK segment flags `stale`; conversions round-trip; a repeated LIMS value is not multiplied across rows.

### Block B — Lag / dead-time features (spec §12–13, eq 32–34)
- `lag.py`, `fir.py`: build effective inputs `ũ` by delay + FIR/first-order shaping (eq 33), and the delay-as-states construction (eq 34).
- **Isolation checks:** step response of eq 33 matches the closed form (compare to `scipy.signal.lsim` of the equivalent FOPTD); FIR weights are non-negative and sum to 1; delay-state shift reproduces a pure delay exactly.

### Block C — Physical kernels (spec §6–9, eq 13–25; §21 eq 46)
- `arrhenius.py` (eq 14–15), `hds_kinetics.py` (Langmuir–Hinshelwood eq 13; reduced power-law eq 16 as a **separate** sanity function), `reactor_balance.py` (eq 19–23 + linearization coefficients), `thermal_balance.py` (eq 24–25), optional `distillation.py` (eq 10–12), `vak.py` (evaluate `ВАК` regressions), `indices.py` (blend indices eq 46).
- **Isolation checks:** Arrhenius derivative eq 15 matches a finite-difference of eq 14; HDS rate decreases with `C_{H2S}` and increases with `T` and `p_{H2}` (sign/monotonicity); reduced-kinetics eq 16 reproduces the LH result in the low-inhibition limit; linearization coefficients (eq 22, 25) match finite-difference Jacobians of eq 19/24 at a test operating point; `vak.py` reproduces a hand-substituted value of one `ВАК` formula; blend index inverts correctly.

### Block D — Discretization & linear identification (spec §10–20, eq 26–45)
- `discretize.py`: ZOH eq 31 (validate against `scipy.signal.cont2discrete`).
- `arx.py`, `fir_ident.py` (eq 35–38), `wls.py` (regularized WLS eq 39–40), `huber.py` (eq 41), `iv.py` (instrumental variables eq 43), `excitation.py` (rank & PE eq 44), `stability.py` (spectral radius / eigenvalue test eq 45).
- **Closed-loop bias is real (eq 42–43):** the data-driven residual/ARX must be estimable with IV, not only OLS. Provide a synthetic **closed-loop** dataset where OLS is provably biased and IV recovers the true gain; test that your IV estimate is within tolerance of truth and OLS is not.
- **Isolation checks:** WLS normal-equation solution matches `numpy.linalg.lstsq` on an unregularized case; Hessian eq 40 is PSD; Huber reduces to least squares for `δ→∞` and downweights an injected outlier; steady-state gain eq 37 matches the simulated step gain; `excitation` correctly flags two near-collinear inputs; `stability` agrees with `numpy.linalg.eigvals`.

### Block E — State estimation & source fusion (spec §22–24, eq 47–52)
- `offset.py` (augmented offset model eq 47), `kalman.py` (predict/update eq 48–50), `gating.py` (χ² innovation gate + `stale` rejection), `fusion.py` (LIMS→PAK→VAK hierarchy, LIMS-anchored scalar bias eq 51–52, variance growth between updates).
- **Isolation checks:** with a linear-Gaussian synthetic system, the Kalman estimate error covariance matches a Monte-Carlo estimate; with no measurement the update reduces to prediction and `P` grows; the innovation gate rejects an injected stuck/biased sensor while accepting valid ones; the hierarchy always prefers a fresh LIMS over PAK/VAK; between LIMS updates the bias variance `P(t)` increases per eq 52.

### Block F — Grey-box predictor, multi-step, uncertainty (spec §26–28, eq 55–60)
- `greybox.py` (eq 55 = kernel + IV/Huber residual + offset), `multistep.py` (eq 56–57: build `𝒪, 𝒯_u, 𝒯_d`, the affine map `Ŷ = 𝒪x̂ + 𝒯_u U + 𝒯_d D`), `uncertainty.py` (stochastic `σ_pred` eq 58 **and** set-based bound eq 59–60).
- **Isolation checks:** the affine map reproduces the recursive multi-step prediction eq 56 exactly on a random stable system (compare `Ŷ` from `𝒯` matrices vs a step-by-step loop); `σ_pred` grows with measurement age `τ` and with distance from the training domain (leverage); the set-based bound eq 60 actually contains simulated worst-case errors; residual monotonicity constraints hold on out-of-domain inputs.

### Block G — Reliability / severity (spec §29–30, eq 61–62)
- `severity.py` (eq 61 robustly standardized proxy aggregation), `deactivation.py` (normalized-temperature drift eq 62, **as a hypothesis, not measured catalyst activity**).
- **Isolation checks:** severity is 0 inside limits and increases monotonically past them; robust standardization uses median/MAD and is insensitive to injected outliers; the deactivation proxy detects an injected slow upward `T_WABT` trend at fixed quality and reports it as a flagged assumption.

### Block H — Blending (spec §4, §21, eq 6–9, 46)
- `blending.py`: sulfur linear-in-mass (eq 7), fraction balance `Σw=1, w≥0`, deviation form eq 8, mixing dynamics eq 9; nonlinear properties via indices eq 46 + additive depressant dose.
- **Isolation checks:** blended sulfur equals mass-weighted average on a 2-component hand example; fractions renormalize/enforce simplex; flash/T95 are **not** blended linearly (a test asserts the index path is used); dose response saturates.

### Block I — Optimization (spec §31–35, eq 63–65)
- `constraints.py` (set `𝒞` §31), `chance.py` (deterministic equivalent eq 63: `Ŝ + z_{1-α}·σ_pred ≤ 10`; plus the set-based check `Ŝ + ε_S⁺ ≤ 10`), `robust.py` (feed-uncertainty counterpart eq 64, box/ellipsoidal `𝒰`, `Γ‖∇_d Ŝ‖`), `objective.py` (eq 65, quality/spec strictly in constraints, **never** in the objective), `solve.py` (convex QP via `cvxpy` when linearized; NLP via `casadi`/ipopt for the LH-nonlinear path; SAA over historical `d^{(s)}` scenarios; start from the current regime; enforce `|Δu|≤Δu_max`).
- **Isolation checks:** on a toy linear instance the chance constraint reduces to a known analytic feasible region; increasing `σ_pred` shrinks the feasible set and pushes `Ŝ` lower (larger back-off); a problem with no feasible point returns *infeasible*, not a silently clipped solution; quality can never be traded for throughput (construct a case where relaxing sulfur would improve `J` and assert the solver refuses it); the robust counterpart eq 64 is feasible-robust against the worst `d∈𝒰` (verify by sampling `𝒰`).

### Block J — Agents & orchestrator (spec §36; ТЗ roles)
- Wrap the blocks as agents with explicit typed interfaces:
  - **Quality agent** → returns `Ŷ`, `σ_pred`/`ε⁺`, `P(y>spec)`, confidence (Blocks C,D,E,F).
  - **Reliability agent** → returns severity class, deactivation flag, tightened `u` bounds (Block G).
  - **Optimization agent** → returns feasible scenarios, metrics, recommended `Δu, w` (Blocks H,I).
  - **Orchestrator** → applies feasibility+freshness+confidence rule eq 66; emits a recommendation **or** a justified refusal; persists the full decision trace to `artifacts/`.
- **Isolation checks:** agents communicate only through their declared interfaces (test that the orchestrator works with each agent replaced by a stub returning fixed values); the refusal branch fires when LIMS is stale (`τ_LIMS>τ_max`), when `σ_pred>σ_max`, or when `𝒞` is empty; every emitted recommendation carries the fields required by the ТЗ report (time+state, risk, action tag old→new, expected effect, constraint checks, confidence, explanation).

### Block K — Identification protocol & validation (spec §37–38, eq 67–69)
- `splits.py` (time splits, leakage guard), `metrics.py` (MAE/RMSE/Bias per horizon eq 67–68, persistence baseline eq 69), `coverage.py` (interval coverage ≈ `1−α`; false/missed spec-violation rates).
- Run the full identification pipeline (delays → kinetics → linear coeffs with IV → excitation prune → filter tuning) and write results + the assumptions register.
- **Isolation checks:** metrics match hand-computed values on a tiny array; the persistence baseline is beaten by the model on a held-out time block *or* the report states honestly that it is not; coverage on held-out LIMS is within tolerance of `1−α` (this is the honesty check on `σ_pred`); one-step vs multi-step vs step-response accuracy are reported separately and not conflated.

### Block L — Demo & entry points
- `cli.py`: run one decision cycle for a given timestamp; `dashboard.py` (optional): show state, prediction with band, constraint status, recommendation/refusal.
- Provide three reproducible demo scenarios: (1) a stable period where the system proposes little/no action; (2) a period with rising sulfur / regime change where it acts; (3) a period with stale/missing/anomalous data where it **refuses** with an explanation.
- **Isolation checks:** each scenario is pinned to explicit timestamps and produces a deterministic, saved output; re-running yields identical artifacts (reproducibility).

---

## 5. Definition of done

- Every block in §4 is INTEGRATED ✅ with its verification report and passing pytest module.
- `pytest` is green; `README.md` reproduces all three demo scenarios from a clean checkout with one documented command sequence.
- `artifacts/assumptions.yaml` lists every non-confirmed coefficient with status and provenance.
- The hard constraint (sulfur ≤ 10 mg/kg via chance constraint eq 63) is provably enforced by tests; quality is never in the objective.
- No temporal leakage (guarded by tests); all sources fused by time with age tracking; refusal path demonstrably works.

## 6. Reporting cadence

After each block, output: (a) the verification report from §0 (eq refs, checks, tolerances, verdict); (b) a one-line note of any deviation from `model_spec.md` and why; (c) the next block you will start. Do not batch multiple unverified blocks. If you are ever tempted to integrate a block because "the prompt says so" — stop, and verify it independently first.
