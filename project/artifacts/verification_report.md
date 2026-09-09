# Verification report

Prompt §0 requires every block to be verified **in isolation before integration**, and §6 requires a report per block: target equations, checks run, tolerances, verdict. This file is generated from the test suite itself (`python -m src.verification_report`), so it cannot drift from what actually runs.

Each block below is `INTEGRATED` only because the listed checks pass. Numerical agreements are stated at the tolerance asserted in the test.

## Block A — Data layer

**Target:** `model_spec.md` §1, §23-25 (eq 1-2, 53-54)

**Modules:** `src/data/units.py`, `src/data/loaders.py`, `src/data/sync.py`, `src/data/staleness.py`, `src/data/tags.py`, `src/data/dataset.py`

**`tests/test_units.py`** — 7 checks

- temperature roundtrip and reference
- mass flow closed form and inverse
- sulfur fraction roundtrip
- molar concentration eq2 hand value
- flow and pressure roundtrips
- partial pressure eq17
- units are vectorised and nan safe

**`tests/test_data_layer.py`** — 17 checks

- detect sentinels finds shared code and spares zero
- detect sentinels ignores single column plateau
- load telemetry drops index columns and masks
- load telemetry edge cases
- paired sheet parser handles blank separator
- sulfur unit normalisation
- asof join known answer and age
- asof join never leaks future
- asof join row index misalignment cannot leak
- asof join repeated value counted once
- asof join turnaround delays availability
- asof join max age masks and empty source
- align telemetry namespaces collide free
- rolling stale flags constant segment only
- rolling stale partial window is not stale
- flatline run length
- robust z and outlier mask resist outliers

**Verdict: INTEGRATED ✅ (24 checks passing)**

---

## Block B — Lag / dead-time features

**Target:** `model_spec.md` §12-13 (eq 32-34), §15 (eq 38)

**Modules:** `src/features/lag.py`, `src/features/fir.py`

**`tests/test_features.py`** — 13 checks

- pure delay exact
- delay states reproduce pure delay
- pure delay rejects negative
- first order step matches closed form
- first order matches scipy zoh discretisation
- foptd delay plus lag
- first order degenerates and guards
- first order handles nan without propagating
- fir weights non negative unit sum
- apply fir hand example and alignment
- fir regressor matrix and gain
- apply fir shorter than kernel
- delay and first order lag commute

**Verdict: INTEGRATED ✅ (13 checks passing)**

---

## Block C — Physical kernels

**Target:** `model_spec.md` §6-9 (eq 13-25), §21 (eq 46), §26 (g_phi)

**Modules:** `src/kernels/arrhenius.py`, `src/kernels/hds_kinetics.py`, `src/kernels/reactor_balance.py`, `src/kernels/thermal_balance.py`, `src/kernels/vak.py`, `src/kernels/sulfur_kernel.py`, `src/blend/indices.py`

**`tests/test_kernels.py`** — 16 checks

- arrhenius two forms agree
- eq15 matches finite difference of eq14
- arrhenius rejects celsius
- fit arrhenius recovers parameters
- lh rate signs and monotonicity
- lh rate edge cases
- analytic partials match finite differences
- eq16 reduced form matches eq13 in low inhibition limit
- eq16 guards
- wabt weights
- eq22 coefficients match numeric jacobian of eq19
- tau C eq23 and stability
- solve steady rejects inconsistent point
- eq25 coefficients match numeric jacobian of eq24
- exothermic runaway is detected not hidden
- coupled A is not diagonal

**`tests/test_vak.py`** — 8 checks

- normalisation of comma and x multiplication
- hand substitution of a real formula
- lims reference becomes an identifier
- truncated formula is disabled not guessed
- repaired formula is documented
- missing symbol yields nan not exception
- all formulas parse or are explicitly flagged
- no arbitrary code execution

**`tests/test_sulfur_kernel.py`** — 6 checks

- prediction is monotone in every argument
- sensitivities have the physically required signs
- fit recovers planted parameters
- fit is robust to injected off spec results
- fit refuses a degenerate dataset
- prediction is nan safe

**Verdict: INTEGRATED ✅ (30 checks passing)**

---

## Block D — Discretisation and linear identification

**Target:** `model_spec.md` §10-20 (eq 26-45)

**Modules:** `src/ident/discretize.py`, `src/ident/arx.py`, `src/ident/fir_ident.py`, `src/ident/wls.py`, `src/ident/huber.py`, `src/ident/iv.py`, `src/ident/excitation.py`, `src/ident/stability.py`, `src/ident/channels.py`

**`tests/test_ident.py`** — 28 checks

- zoh matches scipy cont2discrete
- zoh matches closed form for invertible Ac
- zoh singular Ac still works
- zoh guards
- zoh full splits B and E
- wls matches lstsq unregularised
- wls weights reproduce manual scaling
- wls hessian is psd and ridge shrinks towards theta0
- wls guards
- freshness weights decay with age
- huber reduces to least squares for large delta
- huber downweights an injected outlier
- huber rho psi shapes
- arx steady state gain matches simulated step
- arx gain undefined for integrator
- arx regressors respect delay and mask nan
- cross correlation recovers a planted delay
- ols is biased and iv recovers the true gain closed loop
- ols bias does not vanish with more data
- iv requires enough instruments
- excitation flags near collinear inputs
- excitation accepts well conditioned design
- excitation drops constant column
- spectral radius agrees with numpy
- arx companion and projection
- continuous stability matches eigenvalues
- regularised fir recovers a smooth impulse response
- smoothness matrix shape

**Verdict: INTEGRATED ✅ (28 checks passing)**

---

## Block E — State estimation and source fusion

**Target:** `model_spec.md` §22-24 (eq 47-52)

**Modules:** `src/estimation/offset.py`, `src/estimation/kalman.py`, `src/estimation/gating.py`, `src/estimation/fusion.py`

**`tests/test_estimation.py`** — 17 checks

- augmented model shapes and integrator
- augmentation refuses more offsets than measurements
- augmentation refuses unobservable pair
- no measurement reduces to prediction and P grows
- error covariance matches monte carlo
- gate rejects a biased sensor and accepts a valid one
- stale flag blocks update even when consistent
- joseph form keeps P symmetric psd
- run filter counts rejections
- bias filter anchors to lims and shrinks variance
- bias variance grows between updates eq52
- pak moves the offset less than lims
- hierarchy prefers fresh lims over pak and vak
- stale pak is skipped and old lims falls through
- fuse reports lims pak conflict and resolves by lims
- fuse without any source holds the offset
- gate scalar rules

**Verdict: INTEGRATED ✅ (17 checks passing)**

---

## Block F — Grey-box predictor, multi-step, uncertainty

**Target:** `model_spec.md` §26-28 (eq 55-60)

**Modules:** `src/predict/greybox.py`, `src/predict/multistep.py`, `src/predict/uncertainty.py`, `src/predict/statespace.py`, `src/predict/plant.py`

**`tests/test_predict.py`** — 16 checks

- affine map reproduces the recursion exactly
- affine map is affine in U
- maps shapes and guards
- no input no disturbance is the free response
- sigma pred grows with age offset and horizon
- leverage grows away from the training domain
- z quantile values
- set bound contains simulated worst case errors
- set bound monotone and guards
- empirical half width
- greybox is kernel plus residual plus offset
- sign constraint holds out of domain
- greybox iv path reduces closed loop bias
- greybox guards
- kernel reads shaped inputs not instantaneous ones
- horizon response is delayed and saturating

**`tests/test_statespace.py`** — 9 checks

- realisation matches direct foptd simulation
- initial state converges to the true state
- pure delay channel is exact
- no delay channel is a plain first order lag
- realisation is stable and sized correctly
- multistep maps agree with the realisation
- steady state gain equals the channel gain
- build state space guards
- initial state backfills short history

**`tests/test_plant_replay.py`** — 6 checks

- offset removes a constant model bias
- lab row forecast does not use its own result
- offset variance grows between anchors eq52
- gate rejects an injected spike
- fresh lims wins over pak
- no sources leaves the offset at zero

**Verdict: INTEGRATED ✅ (31 checks passing)**

---

## Block G — Reliability and severity

**Target:** `model_spec.md` §29-30 (eq 61-62)

**Modules:** `src/reliability/severity.py`, `src/reliability/deactivation.py`

**`tests/test_reliability.py`** — 12 checks

- severity is zero inside limits and monotone outside
- severity direction handles a lower limit
- severity index sums and reports the dominant proxy
- severity rejects degenerate scale
- robust reference resists injected outliers
- robust reference survives a constant series
- severity frame evaluation ignores missing columns
- thermal feedback term only fires when unstable
- deactivation detects injected drift at fixed quality
- no drift is not flagged
- bound tightening is capped and explained
- deactivation guards

**Verdict: INTEGRATED ✅ (12 checks passing)**

---

## Block H — Blending

**Target:** `model_spec.md` §4, §21 (eq 6-9, 46)

**Modules:** `src/blend/blending.py`, `src/blend/indices.py`

**`tests/test_blending.py`** — 15 checks

- two component sulfur hand example
- sulfur blend is exactly linear in mass
- simplex enforced and renormalised
- blend sulfur mass guards
- delta form matches finite difference of eq6
- mixer converges to eq6 steady state with tau mix
- index roundtrips
- indexed blend is not linear and is bracketed
- flash and t95 use the index path
- pure component blend returns the component
- unknown property is rejected not blended linearly
- linear properties use eq7
- indexed blend requires simplex
- depressant dose response saturates and is monotone
- depressant applied through blend property

**Verdict: INTEGRATED ✅ (15 checks passing)**

---

## Block I — Constrained robust optimisation

**Target:** `model_spec.md` §31-35 (eq 63-65)

**Modules:** `src/optimize/constraints.py`, `src/optimize/chance.py`, `src/optimize/robust.py`, `src/optimize/objective.py`, `src/optimize/solve.py`

**`tests/test_optimize.py`** — 24 checks

- chance constraint analytic region
- larger sigma forces a lower S hat
- chance min direction and set based check
- saa violation rate
- box support is the worst case over U
- ellipsoid support matches the l2 dual
- robust counterpart is feasible for every sampled d
- uncertainty set guards
- quality can never enter the objective
- objective breakdown and move penalty
- pareto front keeps non dominated
- solver returns a feasible optimum and respects du max
- more uncertainty shrinks the feasible set and lowers S
- infeasible problem is reported not clipped
- quality cannot be traded for throughput
- severity constraint binds
- blend fractions sum to one
- nlp path agrees with the qp on a linear instance
- nlp path handles a nonlinear quality function
- saa reports the violation rate
- bound and spec reporting
- constraint set check is a full audit record
- unidentifiable channels are excluded from manipulated
- model range is narrower than history and labelled

**Verdict: INTEGRATED ✅ (24 checks passing)**

---

## Block J — Agents and orchestrator

**Target:** `model_spec.md` §36 (eq 66); ТЗ §3, §5

**Modules:** `src/agents/contracts.py`, `src/agents/quality_agent.py`, `src/agents/reliability_agent.py`, `src/agents/optimization_agent.py`, `src/agents/orchestrator.py`

**`tests/test_agents.py`** — 10 checks

- orchestrator runs entirely on stubs
- report carries every block the tz requires
- refusal when lims is stale
- fresh pak rescues a stale lims
- refusal when sigma is too large
- refusal when the constraint set is empty
- mitigate is reported as not certifying the spec
- hold when the optimum is the current regime
- decision is json serialisable
- decision is persisted when requested

**Verdict: INTEGRATED ✅ (10 checks passing)**

---

## Block K — Identification protocol and validation

**Target:** `model_spec.md` §37-38 (eq 67-69)

**Modules:** `src/ident/pipeline.py`, `src/validate/splits.py`, `src/validate/metrics.py`, `src/validate/coverage.py`, `src/validate/report.py`, `src/validate/vak_audit.py`

**`tests/test_validate.py`** — 16 checks

- time split is ordered and leak free
- leakage guard fires on a shuffled split
- leakage guard fires on overlapping folds
- embargo removes the boundary window
- rolling origin folds are expanding and leak free
- repeated lims values are not independent samples
- metrics hand computed
- metrics ignore nan and handle empty
- persistence baseline alignment
- horizon table reports skill against persistence
- step response metric reports gain error
- coverage of a correctly specified interval
- coverage detects an overconfident sigma
- one sided coverage matches the chance constraint level
- violation rates hand example
- coverage empty input

**Verdict: INTEGRATED ✅ (16 checks passing)**

---

## Block L — Demonstration and entry points

**Target:** `model_spec.md` prompt Block L; ТЗ §6

**Modules:** `src/system.py`, `src/cli.py`, `src/dashboard.py`, `src/demos.py`, `src/assumptions.py`

**`tests/test_system.py`** — 15 checks

- no temporal leakage in the assembled pipeline
- lims values are never counted twice
- identification used only the training window
- kinetic sensitivity has the physical sign
- demo scenarios produce a complete report
- demo scenarios have their documented outcome
- refusal names its reasons
- mitigation never claims the spec is met
- recommendation respects bounds and rate limits
- only identifiable channels are recommended
- reruns are reproducible
- hard sulfur constraint is enforced on every recommendation
- quality is never in the objective
- assumptions register covers every bound and proxy
- validation report runs and is honest

**Verdict: INTEGRATED ✅ (15 checks passing)**

---

## Total: 235 isolation checks

## Defects found by the isolation loop

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
