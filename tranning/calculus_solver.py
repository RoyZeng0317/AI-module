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
powers, "sin"/"cos"/"tan"/"exp"/"ln"/"log"/"sqrt",
"asin"|"arcsin"/"acos"|"arccos"/"atan"|"arctan"/"sinh"/"cosh"/"tanh", "pi",
"e". Only single-variable expressions in x are supported.

Usage:
    python calculus_solver.py "3x^2 + 5x 的微分"
    python calculus_solver.py "integral of x^2 from 0 to 1"
    python calculus_solver.py "lim(x->0) sin(x)/x"
    python calculus_solver.py "sin(x) 的三階導數"
    python calculus_solver.py "exp(x) 在 x=0 展開到第4階泰勒級數"
    python calculus_solver.py "sin(x) 的5階馬克勞林展開"
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
    # Phase 2 "more function types": inverse trig + hyperbolic, both spellings
    "asin": sp.asin, "arcsin": sp.asin,
    "acos": sp.acos, "arccos": sp.acos,
    "atan": sp.atan, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
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


# Phase 4: multivariable (x, y) parsing — kept entirely separate from
# _LOCAL_DICT/_parse_expr_text above (only used by the partial-derivative/
# gradient patterns below) so the single-variable path's "only x" guarantee
# is untouched; this one allows x and/or y instead.
_LOCAL_DICT_XY = dict(_LOCAL_DICT, y=cg.y)


def _parse_expr_text_xy(text: str):
    text = text.strip()
    if not text:
        return None
    normalized = text.replace("^", "**")
    try:
        expr = parse_expr(normalized, local_dict=_LOCAL_DICT_XY, transformations=_TRANSFORMATIONS)
    except (SyntaxError, TypeError, ValueError, sp.SympifyError):
        return None
    if not expr.free_symbols <= {x, cg.y}:
        return None  # only x/y expressions are supported
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
    # 口語問句："EXPR[, x] 微分後是多少" — variable name前面偶爾會重複提一次
    # (逗號或空白隔開)，跟其他 pattern 一樣用 \s* 吃掉，不特別解析成獨立變數。
    re.compile(r"^(.+?)\s*(?:[,，]\s*(?:對\s*)?x\s*)?微分後(?:是多少|為何|等於多少)?$"),
]


# Phase 3: higher-order derivatives + Taylor/Maclaurin series. Chinese
# ordinals up to 十 (ten) are common in spoken phrasing ("二階導數"); Arabic
# digits also accepted for anything beyond that.
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_order_text(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    return _CN_DIGITS.get(text)


# 中文句型「EXPR 的N階導數」expr 在前、order 在後；d^n/dx^n 記法反過來，
# order 在前、expr 在後 —— group 順序不同，分成兩份清單而不是共用一份。
_NTH_DERIVATIVE_CHINESE_PATTERNS = [
    re.compile(r"^(.+?)\s*的第?\s*([一二三四五六七八九十]|\d+)\s*階(?:導數|微分)$"),
]
_NTH_DERIVATIVE_DDX_PATTERNS = [
    re.compile(r"^d\^(\d+)/dx\^\d+\s*[\(\[]?\s*(.+?)\s*[\)\]]?$", re.IGNORECASE),
]
_TAYLOR_PATTERNS = [
    re.compile(r"^(.+?)\s*在\s*x\s*=\s*(-?\d+)\s*展開到第?\s*(\d+)\s*階(?:的)?泰勒級數$"),
]
_MACLAURIN_PATTERNS = [
    re.compile(r"^(.+?)\s*的第?\s*(\d+)\s*階馬克勞林(?:展開|級數)$"),
]


def _solve_nth_derivative(expr, n):
    try:
        return cg.explain_nth_derivative(expr, n)
    except ValueError as exc:
        raise SolveError(str(exc)) from exc


def _solve_taylor_series(expr, point, order):
    try:
        return cg.explain_taylor_series(expr, point, order)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise SolveError(str(exc)) from exc


# Phase 4: multivariable (x, y) — partial derivative / gradient. "EXPR 對 x
# 的偏微分" has expr first, var second; "∂/∂x(EXPR)" notation is reversed.
_PARTIAL_DERIVATIVE_CHINESE_PATTERNS = [
    re.compile(r"^(.+?)\s*對\s*([xy])\s*(?:的)?偏微分$"),
]
_PARTIAL_DERIVATIVE_NOTATION_PATTERNS = [
    # "∂/∂x" only, never a bare "d/dx" — that notation is already claimed by
    # the single-variable _DERIVATIVE_PATTERNS below (d/dx(sin(x)) etc.) and
    # must keep going through explain_derivative(), not this Phase 4 path.
    re.compile(r"^∂/∂([xy])\s*[\(\[]?\s*(.+?)\s*[\)\]]?$"),
]
_GRADIENT_PATTERNS = [
    re.compile(r"^(.+?)\s*的梯度$"),
    re.compile(r"^gradient of\s*(.+)$", re.IGNORECASE),
]

_VAR_BY_LETTER = {"x": x, "y": cg.y}


def _solve_partial_derivative(expr, var):
    try:
        return cg.explain_partial_derivative(expr, var)
    except ValueError as exc:
        raise SolveError(str(exc)) from exc


def _solve_gradient(expr):
    try:
        return cg.explain_gradient(expr)
    except ValueError as exc:
        raise SolveError(str(exc)) from exc


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
    # 句尾問號/驚嘆號/句號跟算式無關，先去掉再比對，所有 pattern 都是 $ 結尾錨定。
    text = text.rstrip("?？!！。")

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

    for pattern in _NTH_DERIVATIVE_CHINESE_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr_text, order_text = m.groups()
        expr = _parse_expr_text(expr_text)
        n = _parse_order_text(order_text)
        if expr is not None and n is not None:
            return _solve_nth_derivative(expr, n)

    for pattern in _NTH_DERIVATIVE_DDX_PATTERNS:  # order comes first here
        m = pattern.match(text)
        if not m:
            continue
        order_text, expr_text = m.groups()
        expr = _parse_expr_text(expr_text)
        n = _parse_order_text(order_text)
        if expr is not None and n is not None:
            return _solve_nth_derivative(expr, n)

    for pattern in _TAYLOR_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr_text, point_text, order_text = m.groups()
        expr = _parse_expr_text(expr_text)
        if expr is not None:
            return _solve_taylor_series(expr, int(point_text), int(order_text))

    for pattern in _MACLAURIN_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr_text, order_text = m.groups()
        expr = _parse_expr_text(expr_text)
        if expr is not None:
            return _solve_taylor_series(expr, 0, int(order_text))

    for pattern in _PARTIAL_DERIVATIVE_CHINESE_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr_text, var_letter = m.groups()
        expr = _parse_expr_text_xy(expr_text)
        if expr is not None:
            return _solve_partial_derivative(expr, _VAR_BY_LETTER[var_letter])

    for pattern in _PARTIAL_DERIVATIVE_NOTATION_PATTERNS:  # var comes first here
        m = pattern.match(text)
        if not m:
            continue
        var_letter, expr_text = m.groups()
        expr = _parse_expr_text_xy(expr_text)
        if expr is not None:
            return _solve_partial_derivative(expr, _VAR_BY_LETTER[var_letter])

    for pattern in _GRADIENT_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        expr = _parse_expr_text_xy(m.group(1))
        if expr is not None:
            return _solve_gradient(expr)

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
