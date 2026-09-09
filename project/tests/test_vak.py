"""Isolation checks for the ВАК kernel -- model_spec.md §26 (g_phi of eq 55).

Targets: normalisation of the workbook syntax, safe evaluation, hand
substitution of one real formula, and the refusal to evaluate a formula whose
source text is truncated.
"""
import numpy as np
import pytest

from src.data.tags import load_tag_dictionary
from src.kernels import vak
from src.config import load_config


@pytest.fixture(scope="module")
def formulas():
    cfg = load_config()
    td = load_tag_dictionary(cfg.data_path("tags_xlsx"))
    return vak.parse_all(td.vak_raw)


def test_normalisation_of_comma_and_x_multiplication():
    f = vak.parse_formula("AVT6:T", "g", "791,22872 - 5,30294x(F30/(F32+ F30))")
    assert f.ok
    assert "791.22872" in f.expr and "*" in f.expr and "x(" not in f.expr
    assert set(f.symbols) == {"F30", "F32"}
    got = f.evaluate({"F30": np.array([100.0]), "F32": np.array([100.0])})
    assert got[0] == pytest.approx(791.22872 - 5.30294 * 0.5)


def test_hand_substitution_of_a_real_formula(formulas):
    """AVT6:350:D15 = 983.092 + 0.27467*T42 - 0.49014*T48 - 0.32983*F31/F57."""
    f = formulas["AVT6:350:D15"]
    assert f.ok and f.namespace == "AVT"
    assert set(f.symbols) == {"T42", "T48", "F31", "F57"}
    env = {"T42": np.array([285.0]), "T48": np.array([354.0]),
           "F31": np.array([537.0]), "F57": np.array([33.0])}
    expected = (983.092 + 0.27467 * 285.0 - 0.49014 * 354.0
                - 0.32983 * 537.0 / 33.0)
    assert f.evaluate(env)[0] == pytest.approx(expected, rel=1e-12)


def test_lims_reference_becomes_an_identifier(formulas):
    f = formulas["24-2000:GODT:D15"]
    assert f.ok
    assert f.lims_refs == ["LIMS_24_2000_Pipeline_D15"]
    env = {"F22": np.array([10000.0]), "T11": np.array([360.0]),
           "LIMS_24_2000_Pipeline_D15": np.array([830.0])}
    expected = 667.881 + 0.15417 * 830.0 + 0.00005 * 10000.0 + 0.10774 * 360.0
    assert f.evaluate(env)[0] == pytest.approx(expected, rel=1e-12)


def test_truncated_formula_is_disabled_not_guessed(formulas):
    f = formulas["24-2000:GODT:CloudPoint"]
    assert not f.ok
    assert "truncat" in f.note.lower() or "dangling" in f.note.lower()
    with pytest.raises(ValueError):
        f.evaluate({})


def test_repaired_formula_is_documented(formulas):
    f = formulas["AVT6:240-350:CFPP"]
    assert f.ok
    assert f.name in vak.FORMULA_REPAIRS
    assert f.note.startswith("[DOP]")
    # the repair must be the documented ratio, not the broken source text
    env = {k: np.array([1.0]) for k in f.symbols}
    env["F32"], env["F30"] = np.array([2.0]), np.array([2.0])
    got = f.evaluate(env)[0]
    expected = 31.40363 - 0.06784 * 1.0 + 17.411 * 1.0 - 8.11544 * 1.0 \
        - 0.47309 * (1.0 / 4.0)
    assert got == pytest.approx(expected)


def test_missing_symbol_yields_nan_not_exception(formulas):
    f = formulas["AVT6:350:D15"]
    out = f.evaluate({"T42": np.array([1.0, 2.0])})
    assert out.shape == (2,) and np.isnan(out).all()


def test_all_formulas_parse_or_are_explicitly_flagged(formulas):
    for name, f in formulas.items():
        assert f.ok or f.note, f"{name} silently unusable"
    assert sum(f.ok for f in formulas.values()) >= 16


def test_no_arbitrary_code_execution():
    bad = vak.parse_formula("X", "g", "__import__('os').system('echo hi')")
    assert not bad.ok
