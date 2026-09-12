"""Calculus problem generator AND solver — derivative / integral / limit.
Pure sympy symbolic math, NOT a neural model, no checkpoint, nothing to train.

Rule 06 bans external AI models/APIs for anything that "thinks" for this
project — but generating and solving a calculus problem from known
differentiation/integration/limit rules is closed-form math, the same
category as this project's other deterministic algorithms. sinco's own
seq2seq chat model (chats.py) only "knows" prompts it was trained on; this
module is different in kind — every answer is real sympy computation, so it
is always correct and can handle a problem it has never seen before, in
either direction:

  - generate_problem() builds a brand-new random problem + full worked
    solution (used for "出一題微積分"-style quiz requests).
  - explain_derivative()/explain_integral()/explain_limit() take an
    already-parsed sympy expression (built by generate_problem()'s random
    term builders, OR handed in by calculus_solver.py after parsing a
    user-typed expression) and produce that same "problem" shape — so
    solving a problem sinco generated itself and solving a problem the user
    typed in free-form go through the exact same math/explanation code, and
    always agree with each other.

Term builders for derivative/integral (_term_poly/_term_trig/_term_exp) are
sampled *without replacement* (random.Random.sample, not repeated
random.choice) so a single generated problem never picks the same function
family twice — sympy's Add() auto-combines identical terms (e.g. two
independent "poly" picks that both land on x**2 would silently merge into
one term in the displayed function while the step list still explained two
separate terms), so sampling without replacement makes that structurally
impossible instead of just unlikely.

The explain_*() functions work term-by-term via _explain_term_derivative()/
_explain_term_antideriv(), which pattern-match a term (pure power / pure
sin-or-cos / pure exp) for a textbook-style rule name, and fall back to a
generic "sympy 直接計算" line for anything else (products, quotients, nested
functions, ...) — the *answer* is always exactly correct either way (sp.diff
and sp.integrate handle arbitrary elementary expressions), only the
step-by-step wording gets less detailed outside the three known patterns.
explain_limit() is a separate, more general classifier (direct substitution
/ 0-over-0 factor-and-cancel / fallback) — it does not share code with the
four hand-written quiz-limit generators below (_limit_direct/_limit_factor/
_limit_trig/_limit_infinity), which keep their specific textbook phrasing
for the problems sinco poses itself.

explain_nth_derivative()/explain_taylor_series() (higher-order derivatives /
Taylor-Maclaurin series) are solve-only — reachable from calculus_solver.py's
free-form phrasings ("EXPR 的三階導數", "EXPR 在 x=a 展開到 n 階泰勒級數"),
but NOT wired into GENERATORS/generate_problem()'s "出一題微積分" quiz flow;
generate_problem() still only ever returns derivative/integral/limit.

Usage:
    python calculus_generator.py                        # one random-topic problem
    python calculus_generator.py --topic derivative
    python calculus_generator.py --topic integral --count 5
    python calculus_generator.py --topic limit --seed 42 # reproducible
"""

import argparse
import random
import re

import sympy as sp

x = sp.symbols("x")
# Phase 4: second symbol for multivariable calculus (partial derivatives /
# gradient) — explain_partial_derivative()/explain_gradient() near the
# bottom of this file are the only functions that use it. Everything above
# them (the single-variable derivative/integral/limit/series machinery) is
# still hardcoded around x alone, unchanged, so this addition cannot affect
# their behavior.
y = sp.symbols("y")

_POLY_COEFFS = [c for c in range(-9, 10) if c != 0]
_SMALL_COEFFS = [c for c in range(-5, 6) if c != 0]
_INNER_COEFFS = [1, 2, 3]


def _fmt(expr) -> str:
    """Render a sympy expression the way a Traditional-Chinese math text
    would write it: "3*x**2" -> "3x^2". Safe here because every expression
    this module builds/accepts only ever involves the single symbol x, so
    blanket-stripping "*" can never collide two different symbols together.

    sp.E (Euler's number, shows up whenever exp(1) evaluates numerically,
    e.g. a definite integral bound at x=1) prints as a bare "E" —
    indistinguishable from scientific notation at a glance ("4E" reads like
    "4 * 10^?") — so it's lowercased to the conventional "e" first, before
    the blanket "*" strip.
    """
    s = sp.sstr(expr)
    s = s.replace("**", "^")
    s = re.sub(r"\bE\b", "e", s)
    # sympy's own function names (log/asin/acos/atan) aren't what a Chinese
    # math text uses (ln/arcsin/arccos/arctan) — renamed only for display,
    # calculus_solver.py's _LOCAL_DICT still accepts either spelling on input.
    s = re.sub(r"\blog\(", "ln(", s)
    s = re.sub(r"\basin\(", "arcsin(", s)
    s = re.sub(r"\bacos\(", "arccos(", s)
    s = re.sub(r"\batan\(", "arctan(", s)
    s = re.sub(r"\bsqrt\(", "√(", s)
    s = s.replace("*", "")
    return s


# Same log/asin/acos/atan/sqrt renaming as _fmt() above, but as a lookup for
# call sites that only have the bare _match_pure_outer() match `name` string
# in hand (not a full sympy expression to run through _fmt) — e.g. the
# generic "令 u=..., 對 {name}(u) 微分" chain-rule wording.
_DISPLAY_NAME = {"log": "ln", "asin": "arcsin", "acos": "arccos", "atan": "arctan", "sqrt": "√"}


def _inner_str(c: int) -> str:
    """Format an "inner" linear argument c*x the way _fmt would (going
    through sympy so c=1 collapses to "x" instead of a literal "1x") — used
    by call sites that build a step string around cx without first
    constructing the full sympy expression.
    """
    return _fmt(c * x)


def _fmt_limit_value(value) -> str:
    if value == sp.oo:
        return "+∞"
    if value == -sp.oo:
        return "-∞"
    return _fmt(value)


def _fmt_limit_point(point) -> str:
    if point == sp.oo:
        return "∞"
    if point == -sp.oo:
        return "-∞"
    return str(point)


_WILD_C = sp.Wild("c", exclude=[x])
_WILD_K = sp.Wild("k", exclude=[x])
_WILD_U = sp.Wild("u")


# Phase 2 "more function types": log/sqrt (already parseable by
# calculus_solver.py before Phase 2) plus newly-parseable inverse-trig and
# hyperbolic functions. Shared by _match_pure_outer (derivative chain rule)
# and _match_u_substitution (integral u-substitution) below.
_OUTER_FUNCS = (
    (sp.sin, "sin"), (sp.cos, "cos"), (sp.exp, "exp"),
    (sp.log, "log"), (sp.sqrt, "sqrt"),
    (sp.asin, "asin"), (sp.acos, "acos"), (sp.atan, "atan"),
    (sp.sinh, "sinh"), (sp.cosh, "cosh"), (sp.tanh, "tanh"),
)


def _match_pure_outer(term):
    """(name, inner, coeff) if term is *exactly* c * f(inner) for f in
    _OUTER_FUNCS, c a constant not involving x, and inner ANY x-expression —
    linear (c*x, what _is_pure_trig/_is_pure_exp below also catch for the
    original sin/cos/exp trio) or nonlinear (e.g. x**2). Generalizes those
    two older, linear-inner-only matchers so _explain_term_derivative can
    give proper chain-rule wording for a nonlinear inner like sin(x**2)
    instead of falling back to a nameless "sympy 計算" step. Structural
    sp.Wild match, not .has(sp.sin) — same x*sin(x) product-rule caveat as
    the older matchers.
    """
    for func, name in _OUTER_FUNCS:
        m = term.match(_WILD_C * func(_WILD_U))
        if m is not None and x in m[_WILD_U].free_symbols:
            return name, m[_WILD_U], m[_WILD_C]
    return None


def _fraction_parts_if_quotient(term):
    """(numerator, denominator) if term has a genuinely x-dependent
    denominator (needs the quotient rule when differentiating), else None.
    sp.fraction() always returns (term, 1) for anything without an explicit
    negative Pow; only a denominator that actually involves x means the
    quotient rule (rather than the power rule alone) applies."""
    num, den = sp.fraction(term)
    if den == 1 or not den.has(x):
        return None
    return num, den


def _product_parts_if_product_rule(term):
    """(u, v) if term is a Mul with *exactly* two x-dependent factors — e.g.
    x**2*sin(x) — which needs the product rule. A single x-dependent factor
    (c*sin(kx)) is the existing "pure" pattern handled elsewhere instead;
    three or more x-dependent factors fall back to the generic sympy-computed
    step (chained product rule is rare in a textbook quiz setting). Checked
    only after _fraction_parts_if_quotient in the caller, since sympy
    represents a quotient like x/(x+1) internally as Mul(x, (x+1)**-1) — both
    factors "have x", so an unordered check would misclassify a quotient as
    a product.
    """
    if not term.is_Mul:
        return None
    x_factors = [f for f in term.args if f.has(x)]
    if len(x_factors) != 2:
        return None
    return x_factors[0], x_factors[1]


# Restricted to outer functions with a simple, one-line closed-form
# antiderivative (matches _ANTIDERIV_FORMULA_LABELS below plus the original
# sin/cos/exp trio) — log/asin/acos/atan/tanh's antiderivatives need
# integration by parts (e.g. ∫ln(x)dx = x·ln(x)-x), not a clean substitution
# result, so u-substitution wording is intentionally not offered for those;
# sympy still computes the exact right answer either way, just via fallback.
_U_SUB_OUTER_FUNCS = (
    (sp.sin, "sin"), (sp.cos, "cos"), (sp.exp, "exp"),
    (sp.sqrt, "sqrt"), (sp.sinh, "sinh"), (sp.cosh, "cosh"),
)


def _match_u_substitution(term):
    """(name, inner, ratio) if term factors as ratio * inner'(x) * f(inner)
    for f in _U_SUB_OUTER_FUNCS and ratio a constant not involving x — the
    classic u-substitution pattern (e.g. x*sin(x**2) = (1/2)*(2x)*sin(x**2),
    u=x**2, du=2x dx, ratio=1/2). None when no such factorization exists
    (e.g. x*sin(x): inner=x, rest/inner'=x itself still depends on x, so this
    correctly declines rather than guessing — that integral needs integration
    by parts, a technique this module doesn't attempt).
    """
    if not term.is_Mul:
        return None
    factors = term.args
    for i, f in enumerate(factors):
        for func, name in _U_SUB_OUTER_FUNCS:
            m = f.match(func(_WILD_U))
            if m is None:
                continue
            inner = m[_WILD_U]
            if x not in inner.free_symbols:
                continue
            inner_deriv = sp.diff(inner, x)
            if inner_deriv == 0:
                continue
            rest = sp.Mul(*[g for j, g in enumerate(factors) if j != i])
            ratio = sp.simplify(rest / inner_deriv)
            if not ratio.has(x):
                return name, inner, ratio
    return None


def _is_pure_trig(term) -> str | None:
    """"sin" or "cos" if term is *exactly* c * sin(k*x) or c * cos(k*x) (c, k
    constants not involving x — includes the bare function with c=k=1), else
    None. A structural match via sp.Wild, not just "mentions sin somewhere"
    — term.has(sp.sin) would also be True for a product like x*sin(x)
    (needs the product rule, not the plain chain-rule wording this template
    uses) or sin(x)*cos(x), so a loose .has() check would print a
    chain-rule explanation for a term it doesn't actually apply to. Used
    only to pick wording; sp.diff/sp.integrate are correct either way.
    """
    if term.match(_WILD_C * sp.sin(_WILD_K * x)) is not None:
        return "sin"
    if term.match(_WILD_C * sp.cos(_WILD_K * x)) is not None:
        return "cos"
    return None


def _is_pure_exp(term) -> bool:
    """True iff term is *exactly* c * exp(k*x) (see _is_pure_trig — same
    reasoning: term.has(sp.exp) alone would also match x*exp(x)."""
    return term.match(_WILD_C * sp.exp(_WILD_K * x)) is not None


# Phase 2 "more function types" — one-line textbook derivative formulas for
# the 8 outer functions _match_pure_outer added beyond the original sin/cos/
# exp trio (those three keep their own hand-written wording branches below,
# unchanged from Phase 1, since sp.degree()/is_polynomial() already worked
# for them and rewriting risked altering already-tested output).
_DERIVATIVE_FORMULA_LABELS = {
    "log": "(ln x)' = 1/x",
    "sqrt": "(√x)' = 1/(2√x)",
    "asin": "(arcsin x)' = 1/√(1-x²)",
    "acos": "(arccos x)' = -1/√(1-x²)",
    "atan": "(arctan x)' = 1/(1+x²)",
    "sinh": "(sinh x)' = cosh x",
    "cosh": "(cosh x)' = sinh x",
    "tanh": "(tanh x)' = 1-tanh²x",
}

# Antiderivative counterpart — restricted to the 3 new functions with a
# simple, one-line closed-form antiderivative (same reasoning as
# _U_SUB_OUTER_FUNCS above: log/asin/acos/atan/tanh need integration by
# parts, not a clean formula, so they're left to the generic fallback).
_ANTIDERIV_FORMULA_LABELS = {
    "sqrt": "∫√x dx = (2/3)x^(3/2)",
    "sinh": "∫sinh x dx = cosh x",
    "cosh": "∫cosh x dx = sinh x",
}


# ---------------------------------------------------------------------------
# Shared term-level explainers (used by both the random generator and the
# free-form solver in calculus_solver.py)
# ---------------------------------------------------------------------------

def _explain_term_derivative(term) -> tuple:
    deriv = sp.diff(term, x)

    if term.is_polynomial(x):
        step = f"對 {_fmt(term)} 用冪法則 (x^n)' = n·x^(n-1)：({_fmt(term)})' = {_fmt(deriv)}"
        return deriv, step

    quotient = _fraction_parts_if_quotient(term)
    if quotient is not None:
        u, v = quotient
        du, dv = sp.diff(u, x), sp.diff(v, x)
        step = (
            f"對 {_fmt(term)} 用商法則 (u/v)' = (u'v-uv')/v^2，令 u={_fmt(u)}, v={_fmt(v)}："
            f"u'={_fmt(du)}, v'={_fmt(dv)}，({_fmt(term)})' = {_fmt(deriv)}"
        )
        return deriv, step

    product = _product_parts_if_product_rule(term)
    if product is not None:
        u, v = product
        du, dv = sp.diff(u, x), sp.diff(v, x)
        step = (
            f"對 {_fmt(term)} 用乘積律 (uv)' = u'v+uv'，令 u={_fmt(u)}, v={_fmt(v)}："
            f"u'={_fmt(du)}, v'={_fmt(dv)}，({_fmt(term)})' = {_fmt(deriv)}"
        )
        return deriv, step

    outer_match = _match_pure_outer(term)
    if outer_match is not None:
        name, inner, _coeff = outer_match
        if inner.is_polynomial(x) and sp.degree(inner, x) <= 1:
            if name in ("sin", "cos"):
                step = f"對 {_fmt(term)} 用鏈鎖法則微分 {name}(cx)：({_fmt(term)})' = {_fmt(deriv)}"
            elif name == "exp":
                step = f"對 {_fmt(term)} 用指數函數公式 (e^(cx))' = c·e^(cx)：({_fmt(term)})' = {_fmt(deriv)}"
            else:
                label = _DERIVATIVE_FORMULA_LABELS[name]
                step = f"對 {_fmt(term)} 用 {label} 微分公式：({_fmt(term)})' = {_fmt(deriv)}"
        else:
            display_name = _DISPLAY_NAME.get(name, name)
            step = (
                f"對 {_fmt(term)} 用鏈鎖法則，令 u={_fmt(inner)}：對 {display_name}(u) 微分再乘上 "
                f"u'={_fmt(sp.diff(inner, x))}，({_fmt(term)})' = {_fmt(deriv)}"
            )
        return deriv, step

    step = f"對 {_fmt(term)} 直接微分（sympy 計算）：({_fmt(term)})' = {_fmt(deriv)}"
    return deriv, step


def _explain_term_antideriv(term) -> tuple:
    antideriv = sp.integrate(term, x)
    if antideriv.has(sp.Integral):
        raise ValueError(f"sympy 無法求出 {_fmt(term)} 的封閉形式反導函數")

    if term.is_polynomial(x):
        step = f"∫{_fmt(term)} dx 用冪法則 ∫x^n dx = x^(n+1)/(n+1)：= {_fmt(antideriv)}"
        return antideriv, step
    trig_name = _is_pure_trig(term)
    if trig_name:
        step = f"∫{_fmt(term)} dx 用 {trig_name}(cx) 的積分公式：= {_fmt(antideriv)}"
        return antideriv, step
    if _is_pure_exp(term):
        step = f"∫{_fmt(term)} dx 用指數函數積分公式 ∫e^(cx) dx = e^(cx)/c：= {_fmt(antideriv)}"
        return antideriv, step

    outer_match = _match_pure_outer(term)
    if outer_match is not None:
        name, inner, _coeff = outer_match
        if name in _ANTIDERIV_FORMULA_LABELS and inner.is_polynomial(x) and sp.degree(inner, x) <= 1:
            label = _ANTIDERIV_FORMULA_LABELS[name]
            step = f"∫{_fmt(term)} dx 用 {label} 公式，套用線性內函數換算：= {_fmt(antideriv)}"
            return antideriv, step

    u_sub = _match_u_substitution(term)
    if u_sub is not None:
        name, inner, ratio = u_sub
        display_name = _DISPLAY_NAME.get(name, name)
        inner_deriv = sp.diff(inner, x)
        step = (
            f"∫{_fmt(term)} dx 用湊微分（u 代換）：令 u={_fmt(inner)}，du={_fmt(inner_deriv)} dx，"
            f"原式化為 {_fmt(ratio)}∫{display_name}(u) du = {_fmt(antideriv)}"
        )
        return antideriv, step

    step = f"∫{_fmt(term)} dx 直接積分（sympy 計算）：= {_fmt(antideriv)}"
    return antideriv, step


# ---------------------------------------------------------------------------
# Public explain_*() — accept an arbitrary already-built sympy expression
# ---------------------------------------------------------------------------

def explain_derivative(expr) -> dict:
    expr = sp.expand(expr)
    terms = sp.Add.make_args(expr)
    derivs, steps = [], []
    for term in terms:
        d, s = _explain_term_derivative(term)
        derivs.append(d)
        steps.append(s)
    f_prime = sp.Add(*derivs)
    if len(terms) > 1:
        steps.append(f"將各項導數相加：f'(x) = {_fmt(f_prime)}")
    return {
        "topic": "derivative",
        "topic_zh": "微分（導數）",
        "question": f"求下列函數的導數：f(x) = {_fmt(expr)}",
        "steps": steps,
        "answer": f"f'(x) = {_fmt(f_prime)}",
        "func": expr,
        "deriv": f_prime,
    }


def explain_nth_derivative(expr, n: int) -> dict:
    """Phase 3: repeated single differentiation, n times, reusing
    explain_derivative() at *each* order so every one of Phase 1/2's rule
    wordings (power/quotient/product/chain/formula-lookup) is available at
    every order, not just the first — each order's full term-by-term step
    list is kept (prefixed with which order it belongs to) rather than
    collapsed into a single "diff again" line, so a student can see *how*
    each successive derivative was obtained, not just its final form.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"階數必須是正整數，收到 {n!r}")

    current = sp.expand(expr)
    steps = []
    for order in range(1, n + 1):
        sub = explain_derivative(current)
        for s in sub["steps"]:
            steps.append(f"[第 {order} 階] {s}")
        current = sub["deriv"]

    return {
        "topic": "nth_derivative",
        "topic_zh": f"高階微分（第 {n} 階導數）",
        "question": f"求 f(x) = {_fmt(expr)} 的第 {n} 階導數",
        "steps": steps,
        "answer": f"f^({n})(x) = {_fmt(current)}",
        "func": expr,
        "deriv": current,
        "order": n,
    }


def explain_taylor_series(expr, point, order: int) -> dict:
    """Phase 3: Taylor/Maclaurin series expansion. Each step computes one
    coefficient c_k = f^(k)(point)/k! directly (sp.diff(expr, x, k) then
    substitute), the same "one clear formula per step" style as
    explain_limit()'s classifier — not reusing explain_nth_derivative()'s
    term-by-term breakdown, since here only the *value* of each derivative
    at a single point is needed, not a full symbolic derivative expression
    each time.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError(f"展開階數必須是非負整數，收到 {order!r}")

    steps = []
    terms = []
    for k in range(order + 1):
        dk_at_point = sp.diff(expr, x, k).subs(x, point)
        coeff = sp.simplify(dk_at_point / sp.factorial(k))
        if coeff != 0:
            terms.append(coeff if k == 0 else coeff * (x - point) ** k)
        steps.append(
            f"第 {k} 階項：f^({k})({point}) = {_fmt(dk_at_point)}，"
            f"係數 = f^({k})({point})/{k}! = {_fmt(coeff)}"
        )
    poly = sp.expand(sp.Add(*terms)) if point == 0 else sp.Add(*terms)

    kind = "馬克勞林級數" if point == 0 else "泰勒級數"
    return {
        "topic": "taylor_series",
        "topic_zh": kind,
        "question": f"將 f(x) = {_fmt(expr)} 在 x={point} 展開到第 {order} 階的{kind}",
        "steps": steps,
        "answer": f"f(x) ≈ {_fmt(poly)}",
        "func": expr,
        "point": point,
        "order": order,
        "series": poly,
    }


def explain_integral(expr, bounds: tuple | None = None) -> dict:
    expr = sp.expand(expr)
    terms = sp.Add.make_args(expr)
    antiderivs, steps = [], []
    for term in terms:
        F, s = _explain_term_antideriv(term)
        antiderivs.append(F)
        steps.append(s)
    F_expr = sp.Add(*antiderivs)

    if bounds is not None:
        a, b = bounds
        value = sp.simplify(F_expr.subs(x, b) - F_expr.subs(x, a))
        if len(terms) > 1:
            steps.append(f"將各項不定積分相加：F(x) = {_fmt(F_expr)}")
        steps.append(f"代入上下限：F({b}) - F({a}) = {_fmt(value)}")
        return {
            "topic": "integral",
            "topic_zh": "積分（定積分）",
            "question": f"求定積分：∫[{a} 到 {b}] {_fmt(expr)} dx",
            "steps": steps,
            "answer": f"= {_fmt(value)}",
            "func": expr,
            "antideriv": F_expr,
            "definite": True,
            "bounds": (a, b),
            "value": value,
        }

    if len(terms) > 1:
        steps.append(f"將各項不定積分相加並加上常數 C：F(x) = {_fmt(F_expr)} + C")
    return {
        "topic": "integral",
        "topic_zh": "積分（不定積分）",
        "question": f"求不定積分：∫ {_fmt(expr)} dx",
        "steps": steps,
        "answer": f"= {_fmt(F_expr)} + C",
        "func": expr,
        "antideriv": F_expr,
        "definite": False,
        "bounds": None,
        "value": None,
    }


def explain_limit(expr, point) -> dict:
    """General-purpose limit solver for an arbitrary caller-supplied expr
    (used by calculus_solver.py). The final value is always exactly correct
    (sp.limit handles arbitrary elementary expressions); only the step
    wording is best-effort: direct substitution when the expression is
    already defined at `point`, factor-and-cancel when it detects a 0/0
    ratio of polynomials, else a generic "sympy 直接計算" fallback (e.g. for
    trig 0/0 forms or anything sp.fraction can't cleanly split).
    """
    value = sp.limit(expr, x, point)
    value_str = _fmt_limit_value(value)
    point_str = _fmt_limit_point(point)

    steps = None
    if point not in (sp.oo, -sp.oo):
        try:
            num, den = sp.fraction(sp.together(expr))
            indeterminate = sp.simplify(den.subs(x, point)) == 0 and sp.simplify(num.subs(x, point)) == 0
        except (TypeError, ValueError):
            indeterminate = None

        if indeterminate is False:
            steps = [
                f"{_fmt(expr)} 在 x={point} 處有定義",
                f"直接代入：lim(x→{point_str}) {_fmt(expr)} = {_fmt(value)}",
            ]
        elif indeterminate is True:
            common = sp.gcd(sp.factor(num), sp.factor(den))
            if common != 1:
                simplified = sp.cancel(expr)
                steps = [
                    f"代入 x={point}：分子分母皆為 0，屬於 0/0 未定型",
                    f"因式分解並約去公因式 {_fmt(common)}：原式化簡為 {_fmt(simplified)}",
                    f"再代入 x={point}：極限 = {value_str}",
                ]

    if steps is None:
        steps = [f"使用 sympy 直接計算：lim(x→{point_str}) {_fmt(expr)} = {value_str}"]

    return {
        "topic": "limit",
        "topic_zh": "極限",
        "question": f"求極限：lim(x→{point_str}) {_fmt(expr)}",
        "steps": steps,
        "answer": f"= {value_str}",
        "expr": expr,
        "point": point,
        "value": value,
    }


# ---------------------------------------------------------------------------
# Phase 4: multivariable calculus (partial derivatives / gradient), x and y
# only. Deliberately a SEPARATE, self-contained pair of functions — not a
# refactor of _explain_term_derivative() and its helpers above, which are
# hardcoded around the module-level x symbol throughout (_WILD_C = sp.Wild
# ("c", exclude=[x]), sp.diff(term, x), term.is_polynomial(x), ...). Threading
# a variable parameter through all of those to support y as well would touch
# every rule-matcher Phase 1-3 added and tested; instead this reuses only
# sp.diff() itself (which already treats any symbol other than the one being
# differentiated against as a constant automatically — no special-casing
# needed) and writes its own, simpler step wording. _fmt()'s blanket "*"
# strip stays safe with a second symbol in play: "x*y" -> "xy" is still
# unambiguous juxtaposition-multiplication notation, the same as "3x" -> "3x"
# for a single variable — it would only become ambiguous if a symbol were
# ever literally named "xy" itself, which this module never introduces.
# ---------------------------------------------------------------------------

def explain_partial_derivative(expr, var) -> dict:
    """∂expr/∂var (var is x or y), holding the other variable constant.
    expr must only involve x and/or y — no other free symbols supported.
    """
    if var not in (x, y):
        raise ValueError(f"偏微分變數必須是 x 或 y，收到 {var!r}")
    if not expr.free_symbols <= {x, y}:
        raise ValueError("目前的多變數功能只支援 x, y 兩個變數")

    other = y if var is x else x
    deriv = sp.diff(expr, var)
    steps = [
        f"對 {var} 偏微分，將 {other} 視為常數",
        f"逐項微分：∂/∂{var}[{_fmt(expr)}] = {_fmt(deriv)}",
    ]
    return {
        "topic": "partial_derivative",
        "topic_zh": f"偏微分（對 {var}）",
        "question": f"求 f(x,y) = {_fmt(expr)} 對 {var} 的偏微分 ∂f/∂{var}",
        "steps": steps,
        "answer": f"∂f/∂{var} = {_fmt(deriv)}",
        "func": expr,
        "deriv": deriv,
        "var": var,
    }


def explain_gradient(expr) -> dict:
    """∇f = (∂f/∂x, ∂f/∂y) for a two-variable expression — reuses
    explain_partial_derivative() for each component so the wording ("將 y
    視為常數" etc.) stays consistent with the single-partial-derivative path
    above rather than duplicating it.
    """
    if not expr.free_symbols <= {x, y}:
        raise ValueError("目前的多變數功能只支援 x, y 兩個變數")

    px = explain_partial_derivative(expr, x)
    py = explain_partial_derivative(expr, y)
    steps = [f"對 x 的偏微分：{px['answer']}", f"對 y 的偏微分：{py['answer']}"]
    return {
        "topic": "gradient",
        "topic_zh": "梯度",
        "question": f"求 f(x,y) = {_fmt(expr)} 的梯度 ∇f",
        "steps": steps,
        "answer": f"∇f = ({_fmt(px['deriv'])}, {_fmt(py['deriv'])})",
        "func": expr,
        "gradient": (px["deriv"], py["deriv"]),
    }


# ---------------------------------------------------------------------------
# Quiz-only question phrasing variety — the explain_*() functions above stay
# single-phrasing/deterministic on purpose (calculus_solver.py's free-form
# solve path calls them directly with no rng in scope, and re-echoing the
# user's own expression back verbatim is more useful there than a random
# rephrase); only the "出題" quiz generators below (which already thread an
# rng.Random through) rewrite problem["question"] afterwards, picking a
# random wording via that same rng so a given --seed still reproduces the
# exact same problem (see test_same_seed_is_reproducible).
# ---------------------------------------------------------------------------

_DERIVATIVE_QUESTION_TEMPLATES = [
    "求下列函數的導數：f(x) = {expr}",
    "請計算 f(x) = {expr} 的導數",
    "f(x) = {expr}，求 f'(x)",
    "對 f(x) = {expr} 微分",
]
_INTEGRAL_INDEFINITE_QUESTION_TEMPLATES = [
    "求不定積分：∫ {expr} dx",
    "請計算 ∫ {expr} dx",
    "求 {expr} 的不定積分",
]
_INTEGRAL_DEFINITE_QUESTION_TEMPLATES = [
    "求定積分：∫[{a} 到 {b}] {expr} dx",
    "請計算 ∫[{a} 到 {b}] {expr} dx 的值",
    "求 {expr} 從 x={a} 到 x={b} 的定積分",
]
_LIMIT_DIRECT_QUESTION_TEMPLATES = [
    "求極限：lim(x→{a}) {expr}",
    "請計算 lim(x→{a}) {expr}",
    "當 x→{a} 時，求 {expr} 的極限",
]
_LIMIT_FACTOR_QUESTION_TEMPLATES = [
    "求極限：lim(x→{a}) ({num})/({den})",
    "請計算 lim(x→{a}) ({num})/({den})",
]
_LIMIT_SIN_QUESTION_TEMPLATES = ["求極限：lim(x→0) sin({cx})/x", "請計算 lim(x→0) sin({cx})/x"]
_LIMIT_TAN_QUESTION_TEMPLATES = ["求極限：lim(x→0) tan({cx})/x", "請計算 lim(x→0) tan({cx})/x"]
_LIMIT_INFINITY_QUESTION_TEMPLATES = [
    "求極限：lim(x→∞) ({num})/({den})",
    "請計算當 x→∞ 時，({num})/({den}) 的極限",
]


# ---------------------------------------------------------------------------
# Random term builders (quiz generation only)
# ---------------------------------------------------------------------------

def _term_poly(rng: random.Random):
    coeff = rng.choice(_POLY_COEFFS)
    power = rng.randint(1, 4)
    return coeff * x**power


def _term_trig(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    func = sp.sin if rng.random() < 0.5 else sp.cos
    return coeff * func(inner * x)


def _term_exp(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    return coeff * sp.exp(inner * x)


def _term_product(rng: random.Random):
    """u*v term with exactly two x-dependent factors — e.g. x^2*sin(2x) —
    exercising _product_parts_if_product_rule's wording in quiz problems,
    not just calculus_solver.py's free-form path."""
    coeff = rng.choice(_SMALL_COEFFS)
    power = rng.randint(1, 3)
    inner = rng.choice(_INNER_COEFFS)
    func = sp.sin if rng.random() < 0.5 else sp.cos
    return coeff * x**power * func(inner * x)


def _term_chain(rng: random.Random):
    """outer(inner(x)) with a nonlinear inner — e.g. sin(x^2+1) — exercising
    _match_pure_outer's generalized chain-rule wording (the "令 u=..." case)
    in quiz problems; _term_trig/_term_exp above only ever produce a linear
    inner (c*x), which keeps their existing formula-lookup wording."""
    coeff = rng.choice(_SMALL_COEFFS)
    power = rng.choice([2, 3])
    func = rng.choice([sp.sin, sp.cos, sp.exp])
    # sp.expand() (called by explain_derivative) splits exp(a+b) into
    # exp(a)*exp(b), which would print as a confusing extra "exp(2)" factor
    # — sin/cos have no such identity applied by default expand, so only
    # they get an additive constant inside inner.
    const = rng.choice([0, 1, -1, 2]) if func is not sp.exp else 0
    inner = x**power + const
    return coeff * func(inner)


_MORE_FUNCS = (sp.log, sp.sqrt, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh)


def _term_more_functions(rng: random.Random):
    """One of Phase 2's newer function types (log/sqrt/反三角/雙曲) with a
    linear inner c*x — exercises _DERIVATIVE_FORMULA_LABELS' formula-lookup
    wording in quiz problems, not just calculus_solver.py's free-form path."""
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    func = rng.choice(_MORE_FUNCS)
    return coeff * func(inner * x)


_DERIVATIVE_TERM_BUILDERS = [
    _term_poly, _term_trig, _term_exp, _term_product, _term_chain, _term_more_functions,
]


def _term_poly_for_integral(rng: random.Random):
    coeff = rng.choice(_POLY_COEFFS)
    power = rng.randint(0, 4)  # constants are fair game to integrate too
    return coeff * x**power


def _term_u_sub_for_integral(rng: random.Random):
    """coeff * inner'(x) * f(inner(x)) — e.g. 6x*sin(x^2+1) — the classic
    u-substitution shape _match_u_substitution recognizes, so integral
    quizzes also exercise that wording, not just calculus_solver.py's
    free-form path."""
    power = rng.choice([2, 3])
    func = rng.choice([sp.sin, sp.cos, sp.exp, sp.sqrt, sp.sinh, sp.cosh])
    # see _term_chain's docstring above on why exp doesn't get a const offset
    const = rng.choice([0, 1, -1]) if func is not sp.exp else 0
    inner = x**power + const
    coeff = rng.choice(_SMALL_COEFFS)
    return coeff * sp.diff(inner, x) * func(inner)


def _term_more_functions_for_integral(rng: random.Random):
    """sqrt/sinh/cosh with a linear inner c*x — the 3 Phase 2 function types
    with a simple closed-form antiderivative (_ANTIDERIV_FORMULA_LABELS),
    exercising that formula-lookup wording in quiz problems too."""
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    func = rng.choice([sp.sqrt, sp.sinh, sp.cosh])
    return coeff * func(inner * x)


def _generate_derivative(rng: random.Random) -> dict:
    n_terms = rng.randint(1, len(_DERIVATIVE_TERM_BUILDERS))
    builders = rng.sample(_DERIVATIVE_TERM_BUILDERS, n_terms)
    expr = sp.Add(*[b(rng) for b in builders])
    problem = explain_derivative(expr)
    problem["question"] = rng.choice(_DERIVATIVE_QUESTION_TEMPLATES).format(expr=_fmt(problem["func"]))
    return problem


def _generate_integral(rng: random.Random) -> dict:
    builders_pool = [
        _term_poly_for_integral, _term_trig, _term_exp,
        _term_u_sub_for_integral, _term_more_functions_for_integral,
    ]
    n_terms = rng.randint(1, 2)
    builders = rng.sample(builders_pool, n_terms)
    expr = sp.Add(*[b(rng) for b in builders])

    is_definite = rng.random() < 0.5
    bounds = tuple(sorted(rng.sample(range(-3, 4), 2))) if is_definite else None
    problem = explain_integral(expr, bounds=bounds)
    expr_str = _fmt(problem["func"])
    if is_definite:
        a, b = bounds
        problem["question"] = rng.choice(_INTEGRAL_DEFINITE_QUESTION_TEMPLATES).format(expr=expr_str, a=a, b=b)
    else:
        problem["question"] = rng.choice(_INTEGRAL_INDEFINITE_QUESTION_TEMPLATES).format(expr=expr_str)
    return problem


# ---------------------------------------------------------------------------
# Limit (quiz-specific: 4 hand-written pedagogical subtypes)
# ---------------------------------------------------------------------------

def _limit_direct(rng: random.Random) -> dict:
    a = rng.randint(-3, 3)
    coeff1 = rng.choice(_SMALL_COEFFS)
    power = rng.randint(1, 3)
    expr = coeff1 * x**power
    if rng.random() < 0.5:
        coeff2 = rng.choice(_SMALL_COEFFS)
        expr = expr + coeff2 * sp.sin(x)
    value = sp.limit(expr, x, a)
    steps = [
        f"f(x) = {_fmt(expr)} 是多項式／三角函數的組合，處處連續，在 x={a} 處也連續",
        f"連續函數可直接代入：lim(x→{a}) f(x) = f({a}) = {_fmt(value)}",
    ]
    question = rng.choice(_LIMIT_DIRECT_QUESTION_TEMPLATES).format(a=a, expr=_fmt(expr))
    return {
        "topic": "limit",
        "topic_zh": "極限（直接代入）",
        "question": question,
        "steps": steps,
        "answer": f"= {_fmt(value)}",
        "expr": expr,
        "point": a,
        "value": value,
    }


def _limit_factor(rng: random.Random) -> dict:
    a = rng.randint(-3, 3)
    degree = rng.randint(1, 2)
    q_coeffs = [rng.choice(_SMALL_COEFFS) for _ in range(degree)]
    q = sp.Add(*(c * x**i for i, c in enumerate(q_coeffs)))

    denominator = x - a
    numerator = sp.expand(denominator * q)
    expr = numerator / denominator
    value = sp.limit(expr, x, a)

    steps = [
        f"代入 x={a}：分子 = {_fmt(numerator.subs(x, a))}，分母 = {_fmt(denominator.subs(x, a))}，屬於 0/0 未定型",
        f"先因式分解分子：{_fmt(numerator)} = ({_fmt(denominator)})({_fmt(q)})",
        f"約去公因式 ({_fmt(denominator)})：原式化簡為 {_fmt(sp.simplify(q))}",
        f"再代入 x={a}：極限 = {_fmt(value)}",
    ]
    question = rng.choice(_LIMIT_FACTOR_QUESTION_TEMPLATES).format(a=a, num=_fmt(numerator), den=_fmt(denominator))
    return {
        "topic": "limit",
        "topic_zh": "極限（因式分解，0/0 型）",
        "question": question,
        "steps": steps,
        "answer": f"= {_fmt(value)}",
        "expr": expr,
        "point": a,
        "value": value,
    }


def _limit_trig(rng: random.Random) -> dict:
    c = rng.choice(_INNER_COEFFS)
    cx = _inner_str(c)  # "x" when c == 1, else e.g. "3x" — never the literal "1x"
    kind = rng.choice(["sin", "tan"])
    if kind == "sin":
        expr = sp.sin(c * x) / x
        value = sp.limit(expr, x, 0)
        steps = [
            "分子分母代入 x=0 皆為 0，屬於 0/0 未定型",
            f"利用重要極限 lim(x→0) sin(x)/x = 1：把 sin({cx})/x 改寫成 {c}·sin({cx})/({cx})",
            f"當 x→0 時 {cx}→0，所以 sin({cx})/({cx}) → 1",
            f"極限 = {c} · 1 = {_fmt(value)}",
        ]
        question = rng.choice(_LIMIT_SIN_QUESTION_TEMPLATES).format(cx=cx)
    else:
        expr = sp.tan(c * x) / x
        value = sp.limit(expr, x, 0)
        steps = [
            "分子分母代入 x=0 皆為 0，屬於 0/0 未定型",
            "利用 tan(x) = sin(x)/cos(x) 及重要極限 lim(x→0) sin(x)/x = 1：",
            f"tan({cx})/x = {c}·[sin({cx})/({cx})]·[1/cos({cx})]，當 x→0 時 cos({cx})→1",
            f"極限 = {c} · 1 · 1 = {_fmt(value)}",
        ]
        question = rng.choice(_LIMIT_TAN_QUESTION_TEMPLATES).format(cx=cx)

    return {
        "topic": "limit",
        "topic_zh": "極限（重要三角極限）",
        "question": question,
        "steps": steps,
        "answer": f"= {_fmt(value)}",
        "expr": expr,
        "point": 0,
        "value": value,
    }


def _random_poly(rng: random.Random, degree: int):
    lead = rng.choice(_SMALL_COEFFS)
    terms = [lead * x**degree]
    if degree >= 1 and rng.random() < 0.7:
        lower_power = rng.randint(0, degree - 1)
        terms.append(rng.choice(_SMALL_COEFFS) * x**lower_power)
    return sp.Add(*terms), lead


def _limit_infinity(rng: random.Random) -> dict:
    m = rng.randint(1, 3)
    n = rng.randint(1, 3)
    numerator, a_m = _random_poly(rng, m)
    denominator, b_n = _random_poly(rng, n)
    expr = numerator / denominator
    value = sp.limit(expr, x, sp.oo)
    value_str = _fmt_limit_value(value)

    steps = [
        f"分子最高次為 x^{m}（係數 {a_m}），分母最高次為 x^{n}（係數 {b_n}）",
        f"同除以 x^{max(m, n)}，次數較低的項極限皆為 0",
    ]
    if m == n:
        ratio = _fmt(sp.nsimplify(sp.Rational(a_m, b_n)))
        steps.append(f"分子分母同次，只剩最高次項係數之比：極限 = {a_m}/{b_n} = {ratio}")
    elif n > m:
        steps.append("分母次數較高，分子部分同除後趨近於 0：極限 = 0")
    else:
        steps.append(f"分子次數較高，極限發散：極限 = {value_str}")

    question = rng.choice(_LIMIT_INFINITY_QUESTION_TEMPLATES).format(num=_fmt(numerator), den=_fmt(denominator))
    return {
        "topic": "limit",
        "topic_zh": "極限（無窮極限，比較次數）",
        "question": question,
        "steps": steps,
        "answer": f"= {value_str}",
        "expr": expr,
        "point": sp.oo,
        "value": value,
    }


_LIMIT_GENERATORS = [_limit_direct, _limit_factor, _limit_trig, _limit_infinity]


def _generate_limit(rng: random.Random) -> dict:
    gen = rng.choice(_LIMIT_GENERATORS)
    return gen(rng)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "derivative": _generate_derivative,
    "integral": _generate_integral,
    "limit": _generate_limit,
}


def generate_problem(topic: str | None = None, seed: int | None = None) -> dict:
    """Build one random calculus problem + worked solution.

    topic: "derivative" | "integral" | "limit" | "random"/None (pick one at
    random). seed: pass the same int to get the exact same problem back
    (used by tests and by --seed on the CLI); None (default) means a fresh
    random problem every call, which is what the chat-window trigger uses.
    """
    rng = random.Random(seed)
    if topic in (None, "random"):
        topic = rng.choice(list(GENERATORS))
    if topic not in GENERATORS:
        raise ValueError(f"未知主題：{topic!r}，可用選項：{', '.join(GENERATORS)}, random")
    return GENERATORS[topic](rng)


def format_question(problem: dict, heading: str | None = None) -> str:
    """Question-only display (no steps/answer) — used for the "出一題"
    two-step quiz flow: show just the question first, reveal the solution
    later on a follow-up "解題" request (see tools.route_reply()).

    heading defaults to the calculus-specific wording for backward
    compatibility (this module's own generate_problem() callers never pass
    one); word_problem_generator.py/logic_reasoning_generator.py's problems
    are the same dict shape but a different subject, so tools.py passes an
    explicit heading for those instead of the calculus-flavoured default.
    """
    heading = heading or f"sinco 微積分出題：{problem['topic_zh']}"
    lines = [
        f"【{heading}】",
        "",
        f"題目：{problem['question']}",
        "",
        "（輸入「解題」查看詳解與答案）",
    ]
    return "\n".join(lines)


def format_problem(problem: dict, heading: str | None = None) -> str:
    heading = heading or f"sinco 微積分出題：{problem['topic_zh']}"
    lines = [f"【{heading}】", "", f"題目：{problem['question']}", "", "詳解："]
    for i, step in enumerate(problem["steps"], 1):
        lines.append(f"{i}. {step}")
    lines += ["", f"答案：{problem['answer']}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random calculus problem (derivative/integral/limit) with a worked "
                     "solution. Pure sympy — no trained model, no checkpoint."
    )
    parser.add_argument("--topic", choices=["derivative", "integral", "limit", "random"], default="random")
    parser.add_argument("--seed", type=int, default=None, help="reproducible output when given")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    for i in range(args.count):
        seed = None if args.seed is None else args.seed + i
        problem = generate_problem(topic=args.topic, seed=seed)
        print(format_problem(problem))
        print()


if __name__ == "__main__":
    main()
