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
    s = s.replace("*", "")
    return s


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


# ---------------------------------------------------------------------------
# Shared term-level explainers (used by both the random generator and the
# free-form solver in calculus_solver.py)
# ---------------------------------------------------------------------------

def _explain_term_derivative(term) -> tuple:
    deriv = sp.diff(term, x)
    if term.is_polynomial(x):
        step = f"對 {_fmt(term)} 用冪法則 (x^n)' = n·x^(n-1)：({_fmt(term)})' = {_fmt(deriv)}"
        return deriv, step
    trig_name = _is_pure_trig(term)
    if trig_name:
        step = f"對 {_fmt(term)} 用鏈鎖法則微分 {trig_name}(cx)：({_fmt(term)})' = {_fmt(deriv)}"
        return deriv, step
    if _is_pure_exp(term):
        step = f"對 {_fmt(term)} 用指數函數公式 (e^(cx))' = c·e^(cx)：({_fmt(term)})' = {_fmt(deriv)}"
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


_DERIVATIVE_TERM_BUILDERS = [_term_poly, _term_trig, _term_exp]


def _term_poly_for_integral(rng: random.Random):
    coeff = rng.choice(_POLY_COEFFS)
    power = rng.randint(0, 4)  # constants are fair game to integrate too
    return coeff * x**power


def _generate_derivative(rng: random.Random) -> dict:
    n_terms = rng.randint(1, len(_DERIVATIVE_TERM_BUILDERS))
    builders = rng.sample(_DERIVATIVE_TERM_BUILDERS, n_terms)
    expr = sp.Add(*[b(rng) for b in builders])
    return explain_derivative(expr)


def _generate_integral(rng: random.Random) -> dict:
    builders_pool = [_term_poly_for_integral, _term_trig, _term_exp]
    n_terms = rng.randint(1, 2)
    builders = rng.sample(builders_pool, n_terms)
    expr = sp.Add(*[b(rng) for b in builders])

    is_definite = rng.random() < 0.5
    bounds = tuple(sorted(rng.sample(range(-3, 4), 2))) if is_definite else None
    return explain_integral(expr, bounds=bounds)


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
    return {
        "topic": "limit",
        "topic_zh": "極限（直接代入）",
        "question": f"求極限：lim(x→{a}) {_fmt(expr)}",
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
    return {
        "topic": "limit",
        "topic_zh": "極限（因式分解，0/0 型）",
        "question": f"求極限：lim(x→{a}) ({_fmt(numerator)})/({_fmt(denominator)})",
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
        question = f"求極限：lim(x→0) sin({cx})/x"
    else:
        expr = sp.tan(c * x) / x
        value = sp.limit(expr, x, 0)
        steps = [
            "分子分母代入 x=0 皆為 0，屬於 0/0 未定型",
            "利用 tan(x) = sin(x)/cos(x) 及重要極限 lim(x→0) sin(x)/x = 1：",
            f"tan({cx})/x = {c}·[sin({cx})/({cx})]·[1/cos({cx})]，當 x→0 時 cos({cx})→1",
            f"極限 = {c} · 1 · 1 = {_fmt(value)}",
        ]
        question = f"求極限：lim(x→0) tan({cx})/x"

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

    return {
        "topic": "limit",
        "topic_zh": "極限（無窮極限，比較次數）",
        "question": f"求極限：lim(x→∞) ({_fmt(numerator)})/({_fmt(denominator)})",
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


def format_question(problem: dict) -> str:
    """Question-only display (no steps/answer) — used for the "出一題"
    two-step quiz flow: show just the question first, reveal the solution
    later on a follow-up "解題" request (see tools.route_reply()).
    """
    lines = [
        f"【sinco 微積分出題：{problem['topic_zh']}】",
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
