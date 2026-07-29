"""Free-form calculus solver — parses a user-typed request (e.g.
"3x^2+5x 的微分", "∫ x^2 dx", "lim(x->0) sin(x)/x") into a sympy expression +
operation, then hands off to calculus_generator.py's explain_*() functions
for the actual math — so a problem sinco solves for you gets *exactly* the
same step wording as a problem sinco generates on its own (they share the
same term-explanation code).

This is a small regex-based extractor, the same "keyword router, not real
language understanding" honesty as tools.route_reply(): it recognizes a
fixed, documented set of phrasings (see the *_PATTERNS lists below) and
returns None for anything it doesn't recognize, rather than guessing.
Whenever it DOES recognize a request, the final answer is always exactly
correct (real sympy computation) — the only "best effort" part is (a)
whether this module can figure out *what* the user is asking from raw text,
and (b) for expressions outside calculus_generator's three known term
patterns, whether the step wording is a detailed textbook rule or a generic
"sympy 直接計算" fallback line (see calculus_generator.py's docstring).

Supported expression syntax (parsed via sympy.parsing.sympy_parser with
implicit multiplication, so "3x^2" and "3*x**2" both work): "^" or "**" for
powers, "sin"/"cos"/"tan"/"exp"/"ln"/"log"/"sqrt", "pi", "e". Only
single-variable expressions in x are supported.

Usage:
    python calculus_solver.py "3x^2 + 5x 的微分"
    python calculus_solver.py "integral of x^2 from 0 to 1"
    python calculus_solver.py "lim(x->0) sin(x)/x"
"""

import argparse
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

import calculus_generator as cg

x = cg.x

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_LOCAL_DICT = {
    "x": x, "e": sp.E, "pi": sp.pi,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "ln": sp.log, "log": sp.log, "sqrt": sp.sqrt,
}


class SolveError(Exception):
    """Raised when the text WAS recognized as a calculus request but sympy
    could not produce a closed-form result (e.g. an antiderivative with no
    elementary form) — distinct from "not recognized at all" (-> None), so
    callers can tell "I don't understand this request" apart from "I
    understand it but can't solve it" and word the reply accordingly.
    """


def _parse_expr_text(text: str):
    text = text.strip()
    if not text:
        return None
    normalized = text.replace("^", "**")
    try:
        expr = parse_expr(normalized, local_dict=_LOCAL_DICT, transformations=_TRANSFORMATIONS)
    except (SyntaxError, TypeError, ValueError, sp.SympifyError):
        return None
    if not expr.free_symbols <= {x}:
        return None  # only single-variable-x expressions are supported
    return expr


def _parse_limit_point(text: str):
    text = text.strip().lower()
    if text in ("oo", "+oo", "infinity", "+infinity", "∞", "+∞"):
        return sp.oo
    if text in ("-oo", "-infinity", "-∞"):
        return -sp.oo
    try:
        return int(text)
    except ValueError:
        return None


# point comes first in the text: "lim(x->POINT) EXPR" / "lim x->POINT EXPR"
# — two separate patterns (parens mandatory in one, absent in the other)
# rather than one pattern with an optional ")": an optional trailing ")"
# combined with a non-greedy point-group lets the regex engine satisfy the
# match *without* ever consuming the real ")", leaving it stuck inside the
# expression group and producing unparseable text like "oo) 1/x".
_LIMIT_POINT_FIRST_PATTERNS = [
    re.compile(r"^lim\s*\(\s*x\s*(?:→|->|to)\s*(.+?)\s*\)\s*(.+)$", re.IGNORECASE),
    re.compile(r"^lim\s+x\s*(?:→|->|to)\s*(\S+)\s+(.+)$", re.IGNORECASE),
]
# expression comes first: "EXPR 在 x->POINT 的極限" / "EXPR when x->POINT limit"
_LIMIT_EXPR_FIRST_PATTERNS = [
    re.compile(r"^(.+?)\s*(?:在|當)?\s*x\s*(?:→|->|趨近於|to)\s*([^\s的]+)\s*(?:時)?的極限$"),
]

# bounds come first: "∫[a,b] EXPR dx" / "∫ a 到 b EXPR dx"
_INTEGRAL_DEFINITE_BOUNDS_FIRST_PATTERNS = [
    re.compile(r"^∫\s*\[?\s*(-?\d+)\s*(?:到|,|~|-)\s*(-?\d+)\s*\]?\s*(.+?)\s*dx$", re.IGNORECASE),
]
# expression comes first: "EXPR 從 a 到 b 的積分" / "integral of EXPR from a to b"
_INTEGRAL_DEFINITE_EXPR_FIRST_PATTERNS = [
    re.compile(r"^(.+?)\s*(?:從|from)\s*(-?\d+)\s*(?:到|to)\s*(-?\d+)\s*的(?:定)?積分$", re.IGNORECASE),
    re.compile(r"^integral of\s*(.+?)\s*from\s*(-?\d+)\s*to\s*(-?\d+)$", re.IGNORECASE),
]
_INTEGRAL_INDEFINITE_PATTERNS = [
    re.compile(r"^∫\s*(.+?)\s*dx$", re.IGNORECASE),
    re.compile(r"^(.+?)\s*的(?:不定)?積分$"),
    re.compile(r"^integral of\s*(.+)$", re.IGNORECASE),
]

_DERIVATIVE_PATTERNS = [
    re.compile(r"^(?:求|幫我求|算|計算)?\s*(.+?)\s*的(?:微分|導數)$"),
    re.compile(r"^(?:微分|導數|differentiate)\s*[:：]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^derivative of\s*(.+)$", re.IGNORECASE),
    re.compile(r"^d/dx\s*[\(\[]?\s*(.+?)\s*[\)\]]?$", re.IGNORECASE),
]


def _solve_integral(expr, bounds):
    try:
        return cg.explain_integral(expr, bounds=bounds)
    except ValueError as exc:
        raise SolveError(str(exc)) from exc


def parse_and_solve(text: str) -> dict | None:
    """Try each known phrasing in turn; return a calculus_generator-shaped
    problem dict on the first match whose expression parses, else None
    (the message doesn't look like a calculus solve request at all).

    Raises SolveError if the request WAS recognized but sympy could not
    produce a closed-form result — callers should catch this and show a
    friendly message instead of crashing or displaying a bogus answer.
    """
    text = text.strip()
    if not text:
        return None

    for pattern in _LIMIT_POINT_FIRST_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        point = _parse_limit_point(m.group(1))
        expr = _parse_expr_text(m.group(2))
        if point is not None and expr is not None:
            return cg.explain_limit(expr, point)

    for pattern in _LIMIT_EXPR_FIRST_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr = _parse_expr_text(m.group(1))
        point = _parse_limit_point(m.group(2))
        if point is not None and expr is not None:
            return cg.explain_limit(expr, point)

    for pattern in _INTEGRAL_DEFINITE_BOUNDS_FIRST_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        a_text, b_text, expr_text = m.groups()
        expr = _parse_expr_text(expr_text)
        if expr is None:
            continue
        a, b = sorted((int(a_text), int(b_text)))
        return _solve_integral(expr, (a, b))

    for pattern in _INTEGRAL_DEFINITE_EXPR_FIRST_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr_text, a_text, b_text = m.groups()
        expr = _parse_expr_text(expr_text)
        if expr is None:
            continue
        a, b = sorted((int(a_text), int(b_text)))
        return _solve_integral(expr, (a, b))

    for pattern in _INTEGRAL_INDEFINITE_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr = _parse_expr_text(m.group(1))
        if expr is None:
            continue
        return _solve_integral(expr, None)

    for pattern in _DERIVATIVE_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr = _parse_expr_text(m.group(1))
        if expr is None:
            continue
        return cg.explain_derivative(expr)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Parse a free-form calculus request and solve it with sympy (derivative/"
                     "integral/limit). See the module docstring for supported phrasings."
    )
    parser.add_argument("text", help='e.g. "3x^2+5x 的微分", "integral of x^2 from 0 to 1"')
    args = parser.parse_args()

    try:
        problem = parse_and_solve(args.text)
    except SolveError as exc:
        print(f"看得懂這是什麼題目，但 sympy 算不出封閉形式的解：{exc}")
        return

    if problem is None:
        print("看不懂這個算式／請求，請參考模組 docstring 列出的句型。")
        return

    print(cg.format_problem(problem, heading=f"sinco 解題：{problem['topic_zh']}"))


if __name__ == "__main__":
    main()
