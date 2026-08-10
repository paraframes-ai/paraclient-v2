#!/usr/bin/env python3
"""
spreadsheet_schema.py — output format + formula validator for ParaClient's
spreadsheet generator/tutor.

A spreadsheet is {title, cells:[...]} where each cell is a constant
{"ref":"A1","value":x} or a formula {"ref":"B5","formula":"=SUM(B1:B4)"}.
Like circuit ERC and the CAD repair gate, the model's *intent* is checked
against a real evaluator: `spreadsheet_issues` recomputes every formula and
rejects bad refs, circular references, and #DIV/0 — so a served sheet always
computes. The dataset builder uses the same evaluator so training data is
correct by construction.
"""
from __future__ import annotations

import ast
import re

SPREADSHEET_SYS = (
    "You are a spreadsheet generator and tutor. To BUILD a spreadsheet, output "
    "STRICT JSON only (no prose, no markdown):\n"
    '{"title":"...","cells":[{"ref":"A1","value":"Item"},{"ref":"B5","formula":"=SUM(B1:B4)"}]}\n'
    "Rules: A1 notation; a cell has either \"value\" (number or text label) or "
    "\"formula\" (a string starting with =). Use realistic real-world layouts "
    "(budgets, gradebooks, invoices, data tables) with header labels, computed "
    "columns, and totals. Allowed in formulas: cell refs (A1), ranges (A1:A9), "
    "the operators + - * / ( ), and SUM, AVERAGE, MIN, MAX, COUNT, ROUND, ABS. "
    "Every formula must reference existing cells and compute correctly (no "
    "circular references). Output ONLY the JSON object."
)

# --------------------------------------------------------------------------
# A1 <-> (row, col)
# --------------------------------------------------------------------------
def col_to_num(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def num_to_col(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


_REF = re.compile(r"\$?([A-Z]{1,2})\$?(\d+)")
_RANGE = re.compile(r"\$?([A-Z]{1,2})\$?(\d+):\$?([A-Z]{1,2})\$?(\d+)")


def parse_ref(ref: str):
    m = re.fullmatch(r"\$?([A-Z]{1,2})\$?(\d+)", ref)
    if not m:
        raise ValueError(f"bad ref {ref!r}")
    return int(m.group(2)), col_to_num(m.group(1))     # (row, col)


def cell_ref(row: int, col: int) -> str:
    return f"{num_to_col(col)}{row}"


def expand_range(a: str, b: str):
    r1, c1 = parse_ref(a)
    r2, c2 = parse_ref(b)
    return [cell_ref(r, c)
            for r in range(min(r1, r2), max(r1, r2) + 1)
            for c in range(min(c1, c2), max(c1, c2) + 1)]


# --------------------------------------------------------------------------
# safe formula evaluation
# --------------------------------------------------------------------------
_FUNCS = {"r", "v", "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "ROUND", "ABS"}
_ALLOWED = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
            ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.USub, ast.UAdd, ast.List, ast.Tuple)


# Single pass: a range OR a single ref, so refs emitted inside r('A1','A5')
# are never re-scanned (that double-scan was the original bug).
_TOKEN = re.compile(r"(\$?[A-Z]{1,2}\$?\d+):(\$?[A-Z]{1,2}\$?\d+)|(\$?[A-Z]{1,2}\$?\d+)")


def _to_py(formula: str) -> str:
    def rep(m):
        if m.group(1):
            return f"r('{m.group(1).replace('$', '')}','{m.group(2).replace('$', '')}')"
        return f"v('{m.group(3).replace('$', '')}')"
    return _TOKEN.sub(rep, formula)


def _check(tree):
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED):
            raise ValueError(f"disallowed token {type(n).__name__}")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name)
                                            and n.func.id in _FUNCS):
            raise ValueError("bad function")


def _flat(args):
    out = []
    for a in args:
        out.extend(a if isinstance(a, (list, tuple)) else [a])
    return out


def eval_formula(formula: str, get) -> float:
    """Evaluate a =formula; `get(ref)` returns a cell's numeric value or None."""
    tree = ast.parse(_to_py(formula.lstrip("=").strip()), mode="eval")
    _check(tree)

    def v(ref):
        x = get(ref)
        if not isinstance(x, (int, float)):
            raise ValueError(f"ref {ref} is not numeric")
        return x

    def r(a, b):
        return [v(c) for c in expand_range(a, b)]

    ns = {"v": v, "r": r,
          "SUM": lambda *a: sum(_flat(a)),
          "AVERAGE": lambda *a: (sum(_flat(a)) / len(_flat(a))) if _flat(a) else 0,
          "MIN": lambda *a: min(_flat(a)), "MAX": lambda *a: max(_flat(a)),
          "COUNT": lambda *a: len(_flat(a)), "ROUND": lambda x, n=0: round(x, int(n)),
          "ABS": abs}
    return eval(compile(tree, "<f>", "eval"), {"__builtins__": {}}, ns)


def deps_of(formula: str):
    s = formula.lstrip("=")
    refs = set()
    for m in _RANGE.finditer(s):
        refs.update(expand_range(f"{m.group(1)}{m.group(2)}", f"{m.group(3)}{m.group(4)}"))
    for m in _REF.finditer(_RANGE.sub("", s)):
        refs.add(f"{m.group(1)}{m.group(2)}")
    return refs


def evaluate_sheet(cells):
    """Return (computed values dict, [errors]). Resolves formula dependencies
    iteratively; anything left over is a circular/missing-cell reference."""
    computed, formulas, errors = {}, {}, []
    for c in cells:
        if not isinstance(c, dict) or "ref" not in c:
            continue
        ref = c["ref"]
        if isinstance(c.get("formula"), str) and c["formula"].lstrip().startswith("="):
            formulas[ref] = c["formula"]
        elif isinstance(c.get("value"), (int, float)):
            computed[ref] = c["value"]
    remaining = dict(formulas)
    progressed = True
    while remaining and progressed:
        progressed = False
        for ref, f in list(remaining.items()):
            if all(d in computed for d in deps_of(f)):
                try:
                    computed[ref] = eval_formula(f, computed.get)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{ref}: {e}")
                    computed[ref] = 0
                del remaining[ref]
                progressed = True
    for ref in remaining:
        errors.append(f"{ref}: unresolved formula (circular ref or references an "
                      f"empty/text cell)")
    return computed, errors


def spreadsheet_issues(data):
    """(ok, [errors]) — every formula computes, no dup cells. The serving gate."""
    cells = data.get("cells", []) if isinstance(data, dict) else []
    if not cells:
        return True, []
    _vals, errors = evaluate_sheet(cells)
    seen = set()
    for c in cells:
        r = c.get("ref") if isinstance(c, dict) else None
        if r in seen:
            errors.append(f"duplicate cell {r}")
        seen.add(r)
    return (not errors), errors


def guided_schema() -> dict:
    """JSON Schema for guided decoding (`response_format: json_schema`), honoured
    by llama.cpp and vLLM. Guarantees the {title, cells:[{ref, value|formula}]}
    envelope parses; spreadsheet_issues still recomputes every formula, so the
    gateway's rejection sampling targets formula/ref errors rather than malformed
    JSON. `value` may be number or text; extra keys stay open."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "cells": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "value": {"type": ["number", "string"]},
                    "formula": {"type": "string"},
                },
                "required": ["ref"], "additionalProperties": True,
            }},
        },
        "required": ["title", "cells"], "additionalProperties": True,
    }


def clean_spreadsheet(data, _units=None):
    cells = data.get("cells", []) if isinstance(data, dict) else []
    clean = [c for c in cells if isinstance(c, dict) and "ref" in c
             and ("value" in c or isinstance(c.get("formula"), str))]
    title = data.get("title", "Sheet") if isinstance(data, dict) else "Sheet"
    return {"title": title, "cells": clean}
