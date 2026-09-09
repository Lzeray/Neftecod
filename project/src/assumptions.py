"""Assumptions register -- model_spec.md Appendix A; ТЗ §7.

Every coefficient, bound, proxy and data-handling choice that the package does
**not** confirm is written here with its epistemic status and how it was
obtained.  Statuses follow model_spec.md:

    [BALANCE] a consequence of mass/energy conservation -- universal
    [LIT]     a literature correlation for this process -- transferability
              must be checked
    [DERIVED] an algebraic consequence of the equations above
    [DOP]     an assumption whose applicability to this unit is still to be
              confirmed by identification

The rule of ТЗ §4 is enforced verbatim: a historical min/max is never presented
as an industrial limit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src import domain


def _f(x):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def build_register(system) -> dict:
    cfg = system.cfg
    m = system.model
    rep = system.reports
    ds = system.ds

    entries: list[dict] = []

    def add(param, status, source, check, **kw):
        entries.append({"parameter": param, "status": status,
                        "source": source, "check": check, **kw})

    # ---- data handling ------------------------------------------------
    tel = ds.report["telemetry"]
    add("DCS bad-value sentinels", "[DOP]",
        f"data-driven detector: a value repeated >= "
        f"{cfg['cleaning']['sentinel_min_repeats']} times in >= "
        f"{cfg['cleaning']['sentinel_min_columns']} columns and outside that "
        f"column's 0.1-99.9% range. Detected: "
        f"{sorted(set(tel[0]['sentinels']) | set(tel[1]['sentinels']))}",
        "masked cells: "
        f"{tel[0]['cells_masked']} (AVT) + {tel[1]['cells_masked']} (24-2000)")
    add("operating-period mask", "[DOP]",
        ds.report.get("operating_note", ""),
        f"{ds.report.get('operating_rows')} of {ds.report.get('rows')} rows "
        "retained")
    add("LIMS analysis turnaround", "[DOP]",
        "the LIMS workbook carries one timestamp only; availability is taken "
        "as sample time + 4 h so a result is never used before it could exist "
        "(eq 53)",
        "supported by the cross-correlation of the ПАК sulfur against the "
        "laboratory result, which peaks at a ~4 h lag")
    add("LIMS unit row", "[DOP]",
        "the unit row of the LIMS workbook is shifted relative to the "
        "parameter row (50%.T labelled кг/м3, D15 labelled °С); units are "
        "taken from the parameter name instead",
        f"mismatches recorded: {ds.report['lims'].get('unit_row_mismatches')}")
    add("tag dictionary conflicts", "[DOP]",
        "several КИП descriptions disagree with the data attached to them; "
        "the observed physical kind wins and the conflict is recorded",
        f"{len(ds.report['tag_conflicts'])} conflicting tags",
        conflicts=[{"tag": c["key"], "declared": c["declared_kind"],
                    "observed": c["observed_kind"],
                    "description": c["declared"]}
                   for c in ds.report["tag_conflicts"]])

    # ---- channel dynamics (eq 32-34) ----------------------------------
    for k, d in m.channel_dynamics.items():
        add(f"delay/time constant of {k}", "[DOP]",
            "cross-correlation peak of the shaped input against the LIMS "
            "sulfur on the training window (model_spec.md §12, §37.1)",
            f"peak |r| = {abs(d['peak_corr']):.3f} on {d['n_obs']} laboratory "
            f"results; significant = {d['significant']}",
            delay_h=_f(d["delay_h"]), tau_h=_f(d["tau_h"]))

    # ---- kinetics (eq 13-16) ------------------------------------------
    kr = m.kernel.fit_report or {}
    add("activation energy Ea", "[LIT]",
        "fixed at 100 kJ/mol, the centre of the 80-150 kJ/mol range reported "
        "for deep HDS of straight-run gas oil. NOT fitted: on closed-loop data "
        "temperature and sulfur are confounded and a free fit drives Ea to its "
        "lower bound, destroying the kinetic sign the recommendation depends on",
        f"the deployed sensitivity dS/dT is "
        f"{rep.get('step_response', {}).get('dS_dT5_mgkg_per_C', 'n/a')} "
        "mg/kg per degC and is negative as eq 13-15 require",
        value_J_mol=_f(kr.get("Ea_J_mol")))
    add("lumped rate constant k_ref, orders n and a", "[DOP]",
        "non-linear least squares on eq 16 over the training window with a "
        "soft-L1 loss; k_ref absorbs W_cat/V_L and the absolute LHSV, which "
        "the package does not supply",
        f"training MAE {kr.get('MAE_train')} mg/kg on "
        f"{kr.get('n_rows')} laboratory results",
        k_ref=_f(m.kernel.k_ref), n=_f(m.kernel.n), a=_f(m.kernel.a))
    add("product density used in eq 2", "[DOP]",
        "830 kg/m3 assumed for the mg/kg <-> mol/m3 conversion of the product "
        "stream", "consistent with the LIMS D15 of the product point",
        value=_f(m.kernel.rho_kg_m3))
    add("hydrogen mole fraction chi_H2", "[DOP]",
        "0.8 assumed to convert the loop pressure to p_H2 (eq 17); the "
        "package contains no recycle-gas composition",
        "enters only through the kernel; a constant factor is absorbed by "
        "k_ref", value=0.8)
    add("WABT weighting", "[DOP]",
        "1/3 inlet + 2/3 outlet of the instrumented reactor temperatures; the "
        "bed profile is not measured",
        "used consistently in identification and in the sensitivity")

    # ---- residual and instruments (eq 39-43) --------------------------
    add("instrumental variables", "[DOP]",
        rep.get("residual", {}).get("instrument_note", ""),
        rep.get("closed_loop_evidence", {}).get("verdict", ""))
    add("residual sign constraints", "[LIT]",
        "the residual may correct the magnitude of the kinetic effects but "
        "never their direction (model_spec.md §26)",
        f"violations clipped: "
        f"{len(rep.get('residual', {}).get('sign_violations', []))}",
        clipped=rep.get("residual", {}).get("sign_violations", []))
    add("residual shrinkage", "[DOP]",
        "selected on an inner time split of the training window",
        f"chosen scale = {m.filter_params.get('residual_scale')}")

    # ---- filter (eq 48-52) --------------------------------------------
    add("offset filter Q_eta, R_lab and the ПАК switch", "[DOP]",
        "grid search scored by the causal replay MAE on the inner validation "
        "block (model_spec.md §37.5)",
        f"chosen {m.filter_params}; the search rejected the ПАК anchor")
    add("innovation gate", "[DOP]",
        "measurements more than 6 sigma from the prediction are rejected "
        "(model_spec.md §23)",
        f"{rep.get('replay', {}).get('gate_rejected')} of "
        f"{rep.get('replay', {}).get('anchors')} anchors rejected")

    # ---- uncertainty (eq 58-60) ---------------------------------------
    u = rep.get("uncertainty", {})
    add("sigma_pred", "[DOP]",
        u.get("note", "") + " " + u.get("calibrated_on", ""),
        "checked by the held-out interval coverage in "
        "artifacts/validation_report.json",
        sigma_resid=_f(m.sigma_resid), c_tau=_f(m.c_tau))
    add("uncertainty set U for the feed", "[DOP]",
        rep.get("uncertainty_set", {}).get("note", ""),
        "the robust margin of eq 64 is verified by sampling U in "
        "tests/test_optimize.py",
        **{k: v for k, v in rep.get("uncertainty_set", {}).items()
           if k in ("center", "half_widths")})
    for d in domain.UNMEASURED_DISTURBANCES:
        add(f"unmeasured disturbance {d['name']}", d["status"], d["reason"],
            "handled through U (eq 64), never as a fitted regressor")

    # ---- bounds (§31) --------------------------------------------------
    for k, b in m.bounds.items():
        ch = domain.channel(k)
        add(f"operating range of {k}", b.status, b.provenance,
            "NOT an industrial limit; the plant is allowed to sit outside it "
            "and the optimiser may only move towards it at the rate limit",
            lo=_f(b.lo), hi=_f(b.hi), du_max=_f(b.du_max), unit=b.unit,
            controllable=bool(b.identifiable),
            label=(ch.label if ch else ""))

    # ---- severity and deactivation (§29-30) ---------------------------
    for t in m.severity_terms:
        add(f"severity proxy {t.name}", t.status,
            "robust median/MAD standardisation on the training window; the "
            "reference is the 90th percentile of the operating band",
            t.note, ref=_f(t.ref), scale=_f(t.scale), weight=_f(t.weight))
    add("catalyst deactivation", "[DOP]",
        "normalised-temperature drift of eq 62",
        m.deactivation.get("note", ""),
        slope_c_per_1000h=_f(m.deactivation.get("slope_c_per_1000h")),
        significant=bool(m.deactivation.get("significant", False)))

    # ---- objective and economics (§34) --------------------------------
    econ_note = ("Throughput and energy are transparent linear proxies "
                 "normalised by each channel's model range; the package "
                 "contains no economic data (ТЗ §4).")
    add("objective weights c1..c4", "[DOP]",
        f"operator policy, config/config.yaml: {cfg['optimize']['weights']}",
        "quality is never in the objective -- "
        "objective.assert_quality_not_in_objective raises if it is")
    add("throughput and energy proxies", "[DOP]", econ_note,
        "reported alongside every recommendation so the trade-off is visible")
    add("chance level alpha", "[SETTING]",
        f"operator policy: alpha = {cfg['spec']['alpha']}",
        "held-out coverage is reported in artifacts/validation_report.json")
    add("rate limit du_max", "[DOP]",
        f"{cfg['optimize']['du_max_frac']:.0%} of each channel's model range "
        "per decision cycle; no documented plant rate limit was supplied",
        "enforced by the solver and re-checked by constraints.Bound.feasible")

    # ---- blending (§4, §21) -------------------------------------------
    b = cfg.get("blend") or {}
    add("blend component sulfur", "[BALANCE]",
        "the hydrotreated component carries the eq 55 forecast; the "
        "straight-run component carries the measured LIMS sulfur of the HDS "
        "feed point, which is the same material before the reactor",
        "sulfur blends linearly in mass (eq 7) -- verified in "
        "tests/test_blending.py")
    add("blend cold-flow properties and component values", "[DOP]",
        "placeholders in config/config.yaml that exercise the index path of "
        "eq 46; they are not calibrated on this package",
        "must be recalibrated before any operational use",
        components=b.get("components"))
    add("blending indices BI(.)", "[LIT/DOP]",
        "power-law cold-flow index, Wickey-Chittenden flash index, Refutas "
        "viscosity index",
        "inversion and non-linearity verified in tests/test_blending.py")

    # ---- ВАК formulas --------------------------------------------------
    add("ВАК virtual-analyser formulas", "[DOP]",
        "parsed from the workbook (Russian decimal commas, 'x' as the "
        "multiplication sign, one repaired and one truncated formula) and "
        "scored against held-out LIMS",
        "see artifacts/vak_audit.json: a formula is used as a kernel only if "
        "it beats a training-median baseline on the test window")

    return {
        "generated_for_train_end": m.train_end,
        "statuses": {
            "[BALANCE]": "consequence of mass/energy conservation",
            "[LIT]": "literature correlation; transferability to be checked",
            "[DERIVED]": "algebraic consequence of the stated equations",
            "[DOP]": "assumption still to be confirmed by identification",
            "[SETTING]": "operator policy, not a physical quantity",
        },
        "rule": "A historical min/max is never presented as an industrial "
                "limit (ТЗ §4).",
        "entries": entries,
    }


def write_register(system, path: Path | None = None) -> Path:
    reg = build_register(system)
    path = Path(path) if path else system.cfg.artifacts_dir / "assumptions.yaml"
    path.write_text(yaml.safe_dump(reg, allow_unicode=True, sort_keys=False,
                                   default_flow_style=False),
                    encoding="utf-8")
    return path
