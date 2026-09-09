"""Command line entry point.

    python -m src.cli identify              # run the identification protocol
    python -m src.cli validate              # held-out validation report
    python -m src.cli decide  --at "2026-07-06 01:20:00"
    python -m src.cli demo                  # the three pinned scenarios
    python -m src.cli assumptions           # write the assumptions register
    python -m src.cli report                # everything, into artifacts/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import load_config
from src.demos import SCENARIOS


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=_default), encoding="utf-8")


def _print_decision(d) -> None:
    bar = "=" * 78
    print(bar)
    print(f"{d.timestamp}   ACTION: {d.action.upper()}")
    print(bar)
    print(d.headline)
    print()
    print("-- STATE ------------------------------------------------------")
    for k, v in list(d.state_block["key_parameters"].items()):
        print(f"   {k:<14} {v}")
    print("   freshness (h):", {k: v for k, v in
                                d.state_block["freshness_h"].items()})
    print()
    print("-- RISK -------------------------------------------------------")
    rb = d.risk_block
    print(f"   sulfur forecast        {rb['sulfur_forecast_mgkg']} mg/kg")
    print(f"   with back-off (eq 63)  {rb['chance_constrained_value_mgkg']} "
          f"mg/kg   limit {rb['limit_mgkg']}")
    print(f"   P(violation)           {rb['p_violation']}")
    print(f"   set-based bound (60)   {rb['set_based_bound_mgkg']}"
          "  (99% of observed errors; conservative companion of eq 63)")
    print(f"   severity               {rb['severity']} ({rb['severity_class']})")
    print()
    print("-- ACTION -----------------------------------------------------")
    if not d.action_block["changes"]:
        print("   no change proposed")
    for c in d.action_block["changes"]:
        print(f"   {c['label']:<42} {c['current']} -> {c['recommended']} "
              f"{c['unit']}")
    if d.action_block.get("certifies_spec") is False:
        print("   !! this advice does NOT certify the specification")
    print()
    if d.effect_block:
        print("-- EXPECTED EFFECT --------------------------------------------")
        for k, v in d.effect_block.items():
            if k in ("blend", "vs_hold", "response_timing"):
                print(f"   {k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                print(f"   {k:<34} {v}")
        print()
    print("-- CONSTRAINT CHECKS ------------------------------------------")
    if d.constraint_block["gate_eq66"]:
        for f in d.constraint_block["gate_eq66"]:
            print(f"   FAILED: {f}")
    else:
        oks = [c for c in d.constraint_block.get("checks", []) if c.get("ok")]
        bad = [c for c in d.constraint_block.get("checks", []) if not c.get("ok")]
        print(f"   {len(oks)} checks passed, {len(bad)} failed")
        for c in bad[:5]:
            print(f"   FAILED: {c['detail']}")
    print()
    print("-- CONFIDENCE -------------------------------------------------")
    cb = d.confidence_block
    print(f"   {cb['confidence']}  sigma_pred={cb['sigma_pred_mgkg']} mg/kg  "
          f"back-off={cb['backoff_mgkg']} mg/kg")
    for w in cb["warnings"]:
        print(f"   ! {w}")
    print()
    print("-- EXPLANATION ------------------------------------------------")
    for line in _wrap(d.explanation, 74):
        print("   " + line)
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------
def cmd_identify(args) -> int:
    from src.system import build_system
    from src.ident.pipeline import save
    S = build_system(use_cache=not args.no_cache)
    out = S.cfg.artifacts_dir
    save(S.model, out / "identified_model.json")
    _dump(S.ds.report, out / "data_report.json")
    # audit the ВАК formulas against held-out LIMS (§26): a formula is used as
    # a kernel only if it beats a training-median baseline on the test window
    from src.kernels.vak import parse_all
    from src.validate.vak_audit import audit
    va = audit(S.ds.frame, parse_all(S.ds.tagdict.vak_raw),
               S.cfg["time"]["train_end"], S.cfg["time"]["test_end"])
    _dump(va, out / "vak_audit.json")
    from src.ident.pipeline import _slim
    _dump(_slim(S.reports), out / "identification_report.json")
    print(f"identification written to {out}")
    print(f"  operating rows      {S.ds.report['operating_rows']} / "
          f"{S.ds.report['rows']}")
    print(f"  training lab results {S.model.reports['n_lab_train']}")
    print(f"  filter              {S.model.filter_params}")
    print(f"  sigma_resid         {S.model.sigma_resid:.3f} mg/kg")
    print(f"  controllable        "
          f"{[k for k, b in S.model.bounds.items() if b.identifiable]}")
    print(f"  usable ВАК kernels  "
          f"{[k for k, v in va.items() if v.get('usable')]} of {len(va)}")
    return 0


def cmd_validate(args) -> int:
    from src.system import build_system
    from src.validate.report import evaluate
    S = build_system(use_cache=not args.no_cache)
    out = S.cfg.artifacts_dir / "validation_report.json"
    res = evaluate(S, save_to=out)
    print(json.dumps(res["one_step"], indent=2, default=_default))
    print(json.dumps(res["coverage"], indent=2, default=_default))
    print("\nHonesty notes:")
    for n in res["honesty_notes"]:
        print(" -", n)
    print(f"\nfull report: {out}")
    return 0


def cmd_decide(args) -> int:
    from src.system import build_system
    S = build_system(use_cache=not args.no_cache)
    d = S.decide(args.at)
    _print_decision(d)
    print(f"decision trace: {S.cfg.artifacts_dir / 'decisions'}")
    return 0


def cmd_demo(args) -> int:
    from src.system import build_system
    S = build_system(use_cache=not args.no_cache)
    results = {}
    for key, sc in SCENARIOS.items():
        print()
        print("#" * 78)
        print(f"# SCENARIO {key}: {sc['title']}")
        print(f"# expected: {sc['expect']}")
        print("#" * 78)
        d = S.decide(sc["timestamp"])
        _print_decision(d)
        results[key] = {"scenario": sc, "action": d.action,
                        "headline": d.headline}
    _dump(results, S.cfg.artifacts_dir / "demo_summary.json")
    return 0


def cmd_assumptions(args) -> int:
    from src.assumptions import write_register
    from src.system import build_system
    S = build_system(use_cache=not args.no_cache)
    path = write_register(S)
    print(f"assumptions register written to {path}")
    return 0


def cmd_verification(args) -> int:
    from src.verification_report import main as vmain
    return vmain()


def cmd_report(args) -> int:
    for fn in (cmd_identify, cmd_validate, cmd_assumptions, cmd_verification,
               cmd_demo):
        fn(args)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli",
                                 description="AVT -> HDS -> blending diesel "
                                             "quality optimiser")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the parquet/pickle caches and rebuild")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identify").set_defaults(func=cmd_identify)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    d = sub.add_parser("decide")
    d.add_argument("--at", required=True, help="timestamp, e.g. '2026-07-06 01:20:00'")
    d.set_defaults(func=cmd_decide)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    sub.add_parser("assumptions").set_defaults(func=cmd_assumptions)
    sub.add_parser("verification").set_defaults(func=cmd_verification)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    load_config()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
