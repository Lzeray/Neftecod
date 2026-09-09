# AVT → hydrotreater → blending: chance-constrained diesel quality optimiser

A multi-agent decision system for the diesel chain **ЭЛОУ-АВТ-6 → 24-2000 (гидроочистка) → блендинг**.
It implements the model of `prompt/model_spec.md` (Parts I–IX, equations 1–70): a grey-box
quality predictor with an explicit uncertainty budget, a chance-constrained robust
optimiser, and an orchestrator that either issues an explained recommendation or a
justified refusal.

The hard requirement is **product diesel sulfur ≤ 10 mg/kg**. It lives in the constraint
set, never in the objective.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd project
python -m pytest                                    # 237 tests
python -m src.cli report                            # identify + validate + assumptions + demo
```

`python -m src.cli report` runs everything and writes to `artifacts/`. Individual steps:

```bash
python -m src.cli identify                          # §37 identification protocol
python -m src.cli validate                          # §38 held-out validation
python -m src.cli assumptions                       # Appendix A register
python -m src.cli demo                              # the three pinned scenarios
python -m src.cli decide --at "2026-07-12 14:00:00" # one decision cycle
python -m src.dashboard --at "2026-07-12 14:00:00"  # self-contained HTML view
```

Data paths come from `config/config.yaml` (`paths.data_root`, default `..`), which expects

```
<data_root>/data/avt_tags.csv
<data_root>/data/242000_tags.csv
<data_root>/ЛИМСы 01.01.2023 - н.в_ (2).xlsx
<data_root>/Выгрузка ПАК 01.01.2023 - н.в_.xlsx
<data_root>/Теги_хакатон.xlsx
```

Everything is deterministic: one seed, no random splits, parquet/pickle caches keyed by a
hash of the inputs. Re-running produces identical artifacts (`tests/test_system.py::test_reruns_are_reproducible`).
Delete `artifacts/cache/` or pass `--no-cache` to rebuild from scratch (~90 s).

---

## The three demonstration scenarios

All three lie **after** `time.train_end`, so nothing about them was used to fit anything.

| Scenario | Timestamp | Outcome |
|---|---|---|
| Stable period, margin to spec | `2026-07-12 14:00:00` | **RECOMMEND** — the optimiser fine-tunes the regime; all hard constraints hold |
| Operating at the limit | `2025-07-01 00:00:00` | **MITIGATE** — 𝒞 is empty at 95%; the system gives the best risk reduction and states plainly that the limit is *not* certified |
| Stale laboratory data | `2026-04-27 09:00:00` | **REFUSE** — lab result 311 h old, σ_pred above threshold, and the operating point outside the identified domain |

The `RECOMMEND` case is deliberately hard to find, and that is itself the result — see
finding 6.

---

## What the system actually found

These are the results, not the intentions. Held-out window `2025-07-01 → 2026-08-07`,
468 laboratory sulfur results.

**1. The stream sulfur analyser does not track the laboratory.**
`corr(ПАК, ЛИМС) = −0.005` at zero lag over 1458 paired results; after removing the
off-spec tail it rises to **+0.19**, peaking at **+0.37 at a 4 h lag**. The filter tuning
was given ПАК as a *searched option* and rejected it: letting it move the offset made the
forecast worse. Consequently ПАК is not allowed to satisfy the freshness clause of eq 66
either — a source identification rejected cannot rescue a stale lab result.

**2. The closed-loop bias of §18 is visible in the raw data.**
`corr(T_WABT, S_product) = +0.184` — the sign the kinetics forbid. Operators raise the
reactor temperature *because* sulfur is high. A naive regression would therefore recommend
**cooling** the reactor to reduce sulfur.

The package contains **no valid instrument**: there are no external set-point or excitation
tags, and §18 forbids treating an arbitrary lagged tag as one. Rather than pretend
otherwise, the system handles the bias with physics:

* the activation energy is **fixed at a literature value** (100 kJ/mol, `[ЛИТ]`) instead of
  fitted — a free fit drives `Ea` to its lower bound and destroys the kinetic sign;
* every residual coefficient carries a **sign constraint** from eq 13–18. All five HDS-side
  coefficients came out with the physically impossible sign and were clipped to zero.

The deployed temperature sensitivity is **dS/dT = −0.19 mg/kg per °C**, from the kinetics.

**3. The data-driven residual does not survive honest model selection.**
Hyperparameters are chosen on an *inner* time split of the training window. It selected
`residual_scale = 0`: the fitted telemetry gains do not improve out-of-sample forecasts.
The deployed predictor is eq 55 with `r_ψ = 0` — physical kernel plus LIMS-anchored offset.

**4. Prediction quality — reported with error bars, not adjectives.**

| Predictor | MAE, mg/kg | RMSE |
|---|---|---|
| **Model (kernel + offset, eq 55)** | **2.254 ± 0.282** | 6.49 |
| Persistence baseline (eq 69) | 2.303 ± 0.315 | 7.19 |
| Kernel without the offset | 2.702 ± 0.275 | 6.52 |
| Constant at the training median | **1.800 ± 0.262** | 5.93 |

The standard error of the MAE is ~0.28 mg/kg on 468 held-out results, which is larger than
most of the differences between candidate models — so the comparisons are made *paired* on
the shared rows:

* vs. persistence: **−0.050 ± 0.183 mg/kg** — lower on average, **not statistically
  distinguishable** from no difference (t = −0.27);
* vs. a constant at the training median: **+0.454 ± 0.119 mg/kg — significantly worse**
  (t = 3.82).

So the honest verdict is: on one-step forecasts the model is *indistinguishable from
carrying the last lab result forward*, and *measurably worse than a constant*. The product
sulfur of this unit is close to white noise around a slowly moving level; the achievable
skill is in the level — which the LIMS-anchored offset supplies — not in the telemetry.

Over longer horizons the offset does add something: forecasts made 1/2/4/8 lab results
ahead beat persistence by **+9 % to +18 %** MAE. That is the regime where the model helps,
and it is reported separately from the one-step number rather than averaged into it (§38).

**5. The uncertainty estimate is honest.**
The one-sided bound used by eq 63 covers **98.1 %** of held-out results against the 95 % it
claims; the two-sided interval covers **94.2 %** against 95 %. Conservative on the side that
matters for a hard limit, and not optimistic on either.

**6. The plant genuinely cannot certify 10 mg/kg at 95 % confidence — quantified.**
**17.1 %** of held-out laboratory results exceed the limit. With σ_pred ≈ 3.0 mg/kg the
back-off alone is ~5 mg/kg, so from a typical 8–9 mg/kg operating point the chance
constraint is infeasible.

Sweeping the whole held-out window: of **53 640** operating samples that still had a
laboratory result fresher than 72 h, exactly **4** admit a certifying recommendation. On a
302-point uniform scan of the test window the outcome distribution is

| Outcome | Share |
|---|---|
| `mitigate` (𝒞 empty — risk reduction only) | 95.0 % |
| `refuse` (eq 66 blocks any advice) | 5.0 % |
| `recommend` / `hold` (limit certified) | ~0 % |

This is not a modelling artefact. The process really does run at the limit, and no
rate-limited move can buy back a 5 mg/kg back-off.

The system therefore follows §34's **lexicographic** order: stage 1 minimises the
violation, stage 2 optimises economics only if stage 1 reaches zero. When 𝒞 is empty it
returns `mitigate` — the best available risk reduction, labelled `certifies_spec: false`,
with the *attainable* confidence stated (e.g. 67 %). It never silently clips a solution and
never reports a violated constraint as satisfied.

**7. The tag dictionary disagrees with the data in places.**
Five `КИП` descriptions on the 24-2000 side are inconsistent with the signals they are
attached to — e.g. `HDS:T6` is documented as an on-line *sulfur analyser* but reads ~363 °C
and correlates 0.97 with the reactor temperatures; `HDS:W7`, documented as the product
sulfur analyser, correlates 0.04 with the ПАК sulfur and rails at exactly 10.0 (a 0–10 dP
transmitter). The observed physical kind wins and every conflict is recorded in
`artifacts/assumptions.yaml`.

**8. The quality response is delayed, and the report says when.**
`model_spec.md §26` requires the kernel to see the *shaped* inputs of eq 32–33, not the
instantaneous measurements. Identification puts the transport delay at **2.0 h** on the
reactor temperature, **4.5 h** on the feed rate and **5.8 h** (τ = 48 h) on the loop
pressure, so every kernel input is routed through the FOPTD of the channel that drives it.

Consequently the horizon was widened to **7 h** — a shorter one would judge a move on a
window its effect never reaches — and every recommendation carries a `response_timing`
block: transport delay, time constant, time to 63 % of the effect, and the effect actually
realised *within* the horizon. A held +1 °C on the reactor moves the product by
−0.21 mg/kg, none of it in the first 2 h.

**9. Two of eight ВАК formulas earn their place.**
The workbook formulas are not machine-readable as written (Russian decimal commas, `x` as
the multiplication sign, one unbalanced bracket, one truncated expression). After parsing
and repair they are scored against held-out LIMS: only `24-2000:GODT:D15` (+11 % skill) and
`24-2000:GODT:IBP` (+11 %) beat a training-median baseline. `GODT:T90` is off by a factor of
~1400. A first attempt showed `GODT:T95` as useful — that was **leakage**: the formula reads
`LIMS:24-2000.Pipeline.95%.T`, which resolved to its own target. Bound to the *previous*
lab result it fails honestly. See `src/validate/vak_audit.py`.

---

## Architecture

```
data package ──► Block A: loaders / units / as-of sync / staleness
                     │
                     ├─► Block B: FOPTD channel shaping (eq 32–34)
                     ├─► Block C: kinetics, balances, ВАК kernel (eq 13–25, 46)
                     ├─► Block D: ZOH, ARX/FIR, WLS, Huber, IV, PE (eq 26–45)
                     ├─► Block E: Kalman + gating + LIMS anchoring (eq 47–52)
                     ├─► Block F: grey-box, multi-step maps, σ_pred / ε⁺ (eq 55–60)
                     ├─► Block G: severity, deactivation proxy (eq 61–62)
                     ├─► Block H: blending, indices (eq 6–9, 46)
                     └─► Block I: 𝒞, chance, robust, objective, solver (eq 63–65)
                                    │
     ┌──────────────┬───────────────┴───────────────┐
     │ Quality      │ Reliability   │ Optimisation  │   ← Block J, typed interfaces
     │ agent        │ agent         │ agent         │
     └──────────────┴───────┬───────┴───────────────┘
                     Orchestrator — eq 66 gate → recommendation | mitigation | refusal
```

The agents communicate **only** through the dataclasses in `src/agents/contracts.py`.
`tests/test_agents.py` runs the orchestrator with every agent replaced by a stub, which is
what makes the separation real rather than decorative.

### Manipulated variables

Selected from the AVT schematics (`управляемые переменные`) and the `КИП` dictionary, then
filtered by identifiability (§19 → §31): the persistent-excitation test, plus a
**moving-block bootstrap** of each gain. A channel is offered as a control handle only when
its 90 % gain interval excludes zero **or** its sensitivity comes from the physical kernel.

Survivors: `HDS:T5` (reactor outlet temperature), `HDS:P3` (loop pressure → p_H2),
`HDS:F26` (feed rate → LHSV). Excluded with a written reason: `HDS:F15`, `HDS:F2`,
`AVT:T55`, `AVT:T20`, `AVT:P67`, `AVT:F29`.

`model_spec.md §2` is respected: a measured temperature is a **state**, not an input; every
temperature channel is manipulated through its set point and passes through the FOPTD
"set point → actual temperature" block of eq 32–33 before reaching the quality model.

### Disturbances

`crude_density_kgm3` maps to `AVT:D10`, which has **10 non-null rows, all equal to the
bad-value code 240.0** — crude density is effectively unmeasured. `crude_sulfur_wt_pct` and
`crude_water_wt_pct` have no tag at all. They enter only through the robust uncertainty set
𝒰 of eq 64, never as fitted regressors. The HDS **feed** sulfur *is* measured (LIMS
`Mass.Sulfur`, 132 results, % mass → mg/kg by eq 1) and is the box 𝒰 the optimiser is
robust against.

---

## Data handling

* **Time-only synchronisation** (eq 54). Row-index alignment is impossible by construction:
  `sync.asof_join` merges backwards on the timestamp and attaches `__age_h` and `__new`.
  Shuffling the source rows cannot change the result — that is a test.
* **Causal availability** (eq 53). A lab result becomes available at
  `t_sample + turnaround` (4 h, `[ДОП]`), so it is never used before it could exist.
* **Repeated LIMS values are not independent samples** (§25). Every fit and every metric
  goes through `__new` / `splits.unique_measurements`.
* **DCS sentinels** are found data-driven: a value repeated ≥200 times across ≥2 columns and
  outside that column's 0.1–99.9 % range. Detected: `213, 240, 251, 252, 307, 313` and the
  0–10 dP rail. 378 395 + 63 349 cells masked. `0.0` is never treated as a sentinel — it is
  a real shutdown reading.
* **Operating-period mask**: 179 747 of 189 217 rows. Shutdowns are excluded from every fit,
  range and reference.
* **The LIMS unit row is shifted** relative to the parameter row (`50%.T` labelled кг/м3,
  `D15` labelled °С). Units are taken from the parameter name; the mismatch is recorded.

---

## Reproducing the verification

Every block was implemented and verified **in isolation before integration** (prompt §0).
`artifacts/verification_report.md` lists each block, the equations it implements, the checks
run and the verdict. Highlights of the independent references used:

| Block | Independent check |
|---|---|
| eq 31 ZOH | matches `scipy.signal.cont2discrete` to 1e-12, and the closed form `A_c⁻¹(A−I)B_c` |
| eq 33 FOPTD | matches `scipy.signal.lfilter` on the ZOH numerator exactly |
| eq 39–40 WLS | matches `numpy.linalg.lstsq`; Hessian verified PSD |
| eq 41 Huber | reduces to least squares as δ→∞; 10× better than OLS under an injected outlier |
| eq 42–43 IV | on synthetic closed-loop data OLS error is 0.74 and IV 0.008; IV converges, OLS does not |
| eq 45 stability | matches `numpy.linalg.eigvals` |
| eq 22 / 25 | analytic coefficients match central-difference Jacobians of eq 19 / eq 24 |
| eq 48–50 Kalman | reported P matches a 4000-run Monte-Carlo error covariance within 8 % |
| eq 56–57 | the affine map reproduces the step-by-step recursion to 1e-12 on random stable systems |
| eq 59–60 | the set-based bound contains 50 000 sampled worst-case trajectories |
| eq 64 | the box/ellipsoid support functions are tight against 20–50 k samples of 𝒰 |
| eq 26/32–33 | the kernel on shaped inputs differs from the instantaneous one, and a held move shows *no* effect before the identified transport delay |

Three defects were found by that loop and fixed, each with a regression test:

1. Using the IV estimate only as a ridge **anchor** left the closed-loop bias fully intact
   when λ was small. `fit_greybox` now runs a genuine robust IV (Huber-weighted IRLS over eq 43).
2. Clipping a coefficient to satisfy a sign constraint left the **intercept** fitted against
   the unclipped model, biasing every prediction. It is now re-anchored.
3. The state-space delay line was one stage short of `foptd`, so identification and
   prediction disagreed by one sample. Both now agree exactly.

---

## Honest limitations

* No valid instrument exists in the package, so the closed-loop bias (§18) is *contained* by
  physics and sign constraints, not *removed* by identification. The magnitude of
  `dS/dT` rests on a literature `Ea`, not on this plant's data.
* Absolute kinetic constants are not identifiable (no catalyst mass or liquid hold-up is
  supplied); `k_ref` is a lumped constant absorbing `W_cat/V_L` and the absolute LHSV.
* Blend cold-flow properties and component values in `config/config.yaml` are `[ДОП]`
  placeholders that exercise the index path of eq 46. Only the component **sulfur** is
  grounded in measurement.
* The set-based bound of eq 60 is calibrated on the 99 % empirical error range, which §28
  is explicit is *not* a proof of a bound on 𝒲.
* With the back-off applied, the false-alarm rate against the 10 mg/kg limit is 98 % —
  unavoidable when the true violation rate is 17 % and 95 % confidence is demanded. The
  operator-facing lever is `spec.alpha`, and the report always states the *attainable*
  confidence alongside the required one.
* One-step forecasts are not distinguishable from persistence and are significantly worse
  than a constant (finding 4). The value this system adds is **not** point-forecast
  accuracy: it is the calibrated uncertainty, the enforced hard constraint, the sign-correct
  kinetic sensitivity, and the refusal path. A reader looking for a better sulfur predictor
  should read finding 4 first.

Every non-confirmed coefficient, bound and proxy is listed with its status
(`[БАЛАНС] / [ЛИТ] / [ВЫВОД] / [ДОП]`), its source and its check in
`artifacts/assumptions.yaml`. **A historical min/max is never presented as an industrial
limit** — operating ranges are conservative model ranges, explicitly labelled, and the plant
is allowed to sit outside them while the optimiser moves towards them at the rate limit.

---

## Layout

```
project/
  config/config.yaml          paths, horizons, α, weights, seeds, blend definition
  src/
    config.py domain.py system.py cli.py dashboard.py demos.py assumptions.py
    data/      units sync loaders staleness tags dataset
    features/  lag fir
    kernels/   arrhenius hds_kinetics reactor_balance thermal_balance sulfur_kernel vak
    ident/     discretize arx fir_ident wls huber iv excitation stability channels pipeline
    estimation/ offset kalman gating fusion
    predict/   greybox multistep uncertainty statespace plant
    reliability/ severity deactivation
    blend/     blending indices
    optimize/  constraints chance robust objective solve
    agents/    contracts quality_agent reliability_agent optimization_agent orchestrator
    validate/  splits metrics coverage report vak_audit
  tests/                      one module per source module, 237 tests
  artifacts/                  every run's inputs, agent outputs, decisions and reports
```

Each recommendation persists its full decision trace — every agent's output, every
constraint check and the explanation — to `artifacts/decisions/decision_<timestamp>.json`.
