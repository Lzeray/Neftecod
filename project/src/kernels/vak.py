"""ВАК virtual-analyser regressions -- the empirical kernel g_phi of eq (55).

model_spec.md §26: for fractional properties the *kernel* is the plant's own
virtual-analyser regression; the data-driven part only models the residual.

The formulas as delivered in `Теги_хакатон.xlsx` are not machine-readable as
written.  Three defects occur and each is handled explicitly, never silently:

  * Russian decimal commas          -- ``791,22872``      -> ``791.22872``
  * ``x`` (Latin or Cyrillic ``х``) used as the multiplication sign
                                     ``0,52755xT66``      -> ``0.52755*T66``
  * unbalanced / truncated expressions.  ``AVT6:240-350:CFPP`` ends in
    ``(F65/F32+F30))`` -- one closing bracket too many -- and
    ``24-2000:GODT:CloudPoint`` ends in a dangling ``+0.00011`` with no
    operand.  These are *reported as broken* and only used if an explicit,
    documented repair is supplied in ``FORMULA_REPAIRS``.

Symbols are resolved against a namespace: ``AVT6:*`` formulas read AVT tags,
``24-2000:*`` formulas read HDS (unit 24-2000) tags.  That mapping is itself an
assumption and is validated numerically against LIMS in
``src/validate/vak_audit.py`` -- a formula that does not beat a constant
predictor on held-out LIMS data is disabled rather than trusted.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.tags import AVT, HDS

#: Documented repairs of syntactically broken source formulas.
#: key -> (repaired expression, human-readable justification).
FORMULA_REPAIRS: dict[str, tuple[str, str]] = {
    "AVT6:240-350:CFPP": (
        "31.40363 - 0.06784*T33 + 17.411*P67 - 8.11544*P4 - 0.47309*(F65/(F32+F30))",
        "[DOP] Source text `0,47309x(F65/F32+F30))` has one unmatched ')'. The "
        "other AVT formulas use the ratio pattern `F30/(F32+F30)`, so the "
        "intended term is read as F65/(F32+F30).",
    ),
}

#: Formulas whose source text is incomplete; disabled unless repaired.
KNOWN_TRUNCATED = {
    "24-2000:GODT:CloudPoint": (
        "[DOP] Source text ends with a dangling `+0.00011` (no operand) and the "
        "leading term `F22` carries an implicit coefficient of 1 although F22 is "
        "a ~1e4-magnitude flow; the expression is truncated in the workbook. "
        "Disabled -- the residual model carries CloudPoint instead."
    ),
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Constant,
)


@dataclass
class VakFormula:
    name: str
    group: str
    namespace: str                 # AVT | HDS
    source: str                    # raw text from the workbook
    expr: str = ""                 # normalised, evaluable text
    symbols: list[str] = field(default_factory=list)     # bare tag symbols
    lims_refs: list[str] = field(default_factory=list)   # LIMS:... references
    ok: bool = False
    note: str = ""
    _code: object = None

    # ------------------------------------------------------------------
    def evaluate(self, env: dict[str, np.ndarray | float]) -> np.ndarray:
        """Evaluate the regression on a dict of already-namespaced values.

        ``env`` maps *bare* symbol names (``T33``, ``LIMS_24_2000_Pipeline_D15``)
        to scalars or equal-length arrays.  Missing symbols yield NaN rather
        than an exception, so a partially covered timestamp degrades instead of
        crashing.
        """
        if not self.ok:
            raise ValueError(f"formula {self.name} is not evaluable: {self.note}")
        n = 1
        for v in env.values():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                n = max(n, v.shape[0])
        missing = [s for s in self.all_symbols if s not in env]
        if missing:
            return np.full(n, np.nan)
        local = {s: np.asarray(env[s], dtype=float) for s in self.all_symbols}
        with np.errstate(divide="ignore", invalid="ignore"):
            out = eval(self._code, {"__builtins__": {}}, local)  # noqa: S307
        return np.broadcast_to(np.asarray(out, dtype=float), (n,)).copy()

    @property
    def all_symbols(self) -> list[str]:
        return list(self.symbols) + list(self.lims_refs)


# ======================================================================
def _normalise(text: str) -> str:
    """Russian decimal commas + 'x'/'х' multiplication -> Python syntax."""
    s = text.replace(" ", " ").strip()
    # decimal comma: digit , digit  ->  digit . digit
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    # 'x' or Cyrillic 'х' between a number/paren and an identifier/paren
    s = re.sub(r"(?<=[\d\)])\s*[xхх]\s*(?=[A-Za-z\(])", "*", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _lims_placeholder(s: str) -> tuple[str, list[str]]:
    """Replace ``LIMS:24-2000.Pipeline.95%.T`` by a valid identifier."""
    refs: list[str] = []

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        ident = "LIMS_" + re.sub(r"[^0-9A-Za-z]+", "_", raw[len("LIMS:"):]).strip("_")
        if ident not in refs:
            refs.append(ident)
        return ident

    s = re.sub(r"LIMS:[0-9A-Za-z\-\.%:]+", repl, s)
    return s, refs


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax element {type(node).__name__}")


def parse_formula(name: str, group: str, source: str) -> VakFormula:
    ns = HDS if name.startswith("24-2000") else AVT
    f = VakFormula(name=name, group=group, namespace=ns, source=source)

    if name in KNOWN_TRUNCATED and name not in FORMULA_REPAIRS:
        f.ok = False
        f.note = KNOWN_TRUNCATED[name]
        return f

    if name in FORMULA_REPAIRS:
        text, why = FORMULA_REPAIRS[name]
        f.note = why
    else:
        text = _normalise(source)

    text, lims_refs = _lims_placeholder(text)
    f.expr, f.lims_refs = text, lims_refs

    try:
        tree = ast.parse(text, mode="eval")
        _validate_ast(tree)
    except (SyntaxError, ValueError) as exc:
        f.ok = False
        f.note = (f.note + " | " if f.note else "") + f"unparseable: {exc}"
        return f

    syms = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    f.symbols = [s for s in syms if s not in lims_refs]
    f._code = compile(tree, filename=f"<vak:{name}>", mode="eval")
    f.ok = True
    return f


def parse_all(vak_raw: list[dict]) -> dict[str, VakFormula]:
    return {r["name"]: parse_formula(r["name"], r["group"], r["formula"])
            for r in vak_raw}


# ======================================================================
def build_env(formula: VakFormula, frames: dict[str, pd.DataFrame],
              lims_wide: pd.DataFrame | None = None,
              lims_map: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """Assemble the evaluation environment for one formula.

    ``frames`` maps the namespace (``AVT``/``HDS``) to a *time-aligned* frame
    whose columns are the bare short tag names.  ``lims_wide`` supplies the
    as-of merged LIMS columns referenced by ``LIMS:...`` symbols, resolved
    through ``lims_map`` (identifier -> column of ``lims_wide``).
    """
    df = frames[formula.namespace]
    env: dict[str, np.ndarray] = {}
    for s in formula.symbols:
        if s in df.columns:
            env[s] = df[s].to_numpy(dtype=float)
    if formula.lims_refs and lims_wide is not None and lims_map is not None:
        for ident in formula.lims_refs:
            col = lims_map.get(ident)
            if col is not None and col in lims_wide.columns:
                env[ident] = lims_wide[col].to_numpy(dtype=float)
    return env
