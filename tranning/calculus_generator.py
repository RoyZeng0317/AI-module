"""Random calculus problem generator — derivative / integral / limit, with a
worked (step-by-step) solution. Pure sympy symbolic math, NOT a neural model.

Rule 06 bans external AI models/APIs for anything that "thinks" for this
project — but generating and solving a calculus problem from known
differentiation/integration/limit rules is closed-form math, the same
category as kicad_dataset_convert.py's S-expression parser or
dataset_import.py's file-format conversions: a deterministic algorithm, not
a model. There is nothing to train and no checkpoint to load, so (unlike
chats.py's seq2seq model, which only "knows" prompts it was trained on)
this can generate a genuinely new, never-seen-before problem every call.

Wired into the chat window the same way tools.route_reply() already handles
weather/search: a "出一題微積分/微分/積分/極限" (or English "quiz me on
calculus/derivative/integral/limit") -shaped message is detected and routed
here directly, bypassing the trained seq2seq model entirely — sinco's own
model is a small memorization network (see chats.py's docstring) and cannot
reliably produce *novel* problems, so this is real computation instead of a
guess, exactly like the weather/search live lookups.

Term generators for derivative/integral are sampled *without replacement*
(random.Random.sample, not repeated random.choice) so a single problem never
picks the same function family twice — that keeps the printed step list in
lock-step with the printed question (sympy's Add() auto-combines identical
terms, e.g. two independent "poly" picks that both land on x**2 would merge
into one term in the displayed function while the step list still explained
two separate terms; sampling without replacement makes that structurally
impossible instead of just unlikely).

Every returned problem also carries the raw sympy objects it was built from
(e.g. "func"/"deriv", "expr"/"point"/"value") alongside the formatted text —
not used by format_problem(), but lets tests independently re-derive the
correct answer with sp.diff/sp.integrate/sp.limit instead of trusting the
generator's own arithmetic.

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
    this module builds only ever involves the single symbol x, so blanket-
    stripping "*" can never collide two different symbols together.

    sp.E (Euler's number, shows up whenever exp(1) evaluates numerically,
    e.g. a definite integral bound at x=1) prints as a bare "E" — indistinguishable
    from scientific notation at a glance ("4E" reads like "4 * 10^?") — so it's
    lowercased to the conventional "e" first, before the blanket "*" strip.
    """
    s = sp.sstr(expr)
    s = s.replace("**", "^")
    s = re.sub(r"\bE\b", "e", s)
    s = s.replace("*", "")
    return s


def _inner_str(c: int) -> str:
    """Format an "inner" linear argument c*x the way _fmt would (going
    through sympy so c=1 collapses to "x" instead of the literal "1x") —
    used by call sites that build a step string around cx without first
    constructing the full sympy expression.
    """
    return _fmt(c * x)


def _fmt_limit_value(value) -> str:
    if value == sp.oo:
        return "+∞"
    if value == -sp.oo:
        return "-∞"
    return _fmt(value)


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------

def _term_poly_derivative(rng: random.Random):
    coeff = rng.choice(_POLY_COEFFS)
    power = rng.randint(1, 4)
    expr = coeff * x**power
    deriv = sp.diff(expr, x)
    step = f"對 {_fmt(expr)} 用冪法則 (x^n)' = n·x^(n-1)：({_fmt(expr)})' = {_fmt(deriv)}"
    return expr, deriv, step


def _term_trig_derivative(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    is_sin = rng.random() < 0.5
    func = sp.sin if is_sin else sp.cos
    expr = coeff * func(inner * x)
    deriv = sp.diff(expr, x)
    name = "sin" if is_sin else "cos"
    step = f"對 {_fmt(expr)} 用鏈鎖法則微分 {name}({inner}x)：({_fmt(expr)})' = {_fmt(deriv)}"
    return expr, deriv, step


def _term_exp_derivative(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    expr = coeff * sp.exp(inner * x)
    deriv = sp.diff(expr, x)
    step = f"對 {_fmt(expr)} 用指數函數公式 (e^(cx))' = c·e^(cx)：({_fmt(expr)})' = {_fmt(deriv)}"
    return expr, deriv, step


_DERIVATIVE_TERM_GENERATORS = [_term_poly_derivative, _term_trig_derivative, _term_exp_derivative]


def _generate_derivative(rng: random.Random) -> dict:
    n_terms = rng.randint(1, len(_DERIVATIVE_TERM_GENERATORS))
    generators = rng.sample(_DERIVATIVE_TERM_GENERATORS, n_terms)

    exprs, derivs, steps = [], [], []
    for gen in generators:
        e, d, s = gen(rng)
        exprs.append(e)
        derivs.append(d)
        steps.append(s)

    f_expr = sp.Add(*exprs)
    f_prime = sp.Add(*derivs)
    if n_terms > 1:
        steps.append(f"將各項導數相加：f'(x) = {_fmt(f_prime)}")

    return {
        "topic": "derivative",
        "topic_zh": "微分（導數）",
        "question": f"求下列函數的導數：f(x) = {_fmt(f_expr)}",
        "steps": steps,
        "answer": f"f'(x) = {_fmt(f_prime)}",
        "func": f_expr,
        "deriv": f_prime,
    }


# ---------------------------------------------------------------------------
# Integral
# ---------------------------------------------------------------------------

def _term_poly_antideriv(rng: random.Random):
    coeff = rng.choice(_POLY_COEFFS)
    power = rng.randint(0, 4)
    expr = coeff * x**power
    antideriv = sp.integrate(expr, x)
    step = f"∫{_fmt(expr)} dx 用冪法則 ∫x^n dx = x^(n+1)/(n+1)：= {_fmt(antideriv)}"
    return expr, antideriv, step


def _term_trig_antideriv(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    is_sin = rng.random() < 0.5
    func = sp.sin if is_sin else sp.cos
    expr = coeff * func(inner * x)
    antideriv = sp.integrate(expr, x)
    name = "sin" if is_sin else "cos"
    step = f"∫{_fmt(expr)} dx 用 {name}(cx) 的積分公式：= {_fmt(antideriv)}"
    return expr, antideriv, step


def _term_exp_antideriv(rng: random.Random):
    coeff = rng.choice(_SMALL_COEFFS)
    inner = rng.choice(_INNER_COEFFS)
    expr = coeff * sp.exp(inner * x)
    antideriv = sp.integrate(expr, x)
    step = f"∫{_fmt(expr)} dx 用指數函數積分公式 ∫e^(cx) dx = e^(cx)/c：= {_fmt(antideriv)}"
    return expr, antideriv, step


_INTEGRAL_TERM_GENERATORS = [_term_poly_antideriv, _term_trig_antideriv, _term_exp_antideriv]


def _generate_integral(rng: random.Random) -> dict:
    n_terms = rng.randint(1, 2)
    generators = rng.sample(_INTEGRAL_TERM_GENERATORS, n_terms)

    exprs, antiderivs, steps = [], [], []
    for gen in generators:
        e, F, s = gen(rng)
        exprs.append(e)
        antiderivs.append(F)
        steps.append(s)

    f_expr = sp.Add(*exprs)
    F_expr = sp.Add(*antiderivs)

    is_definite = rng.random() < 0.5
    if is_definite:
        a, b = sorted(rng.sample(range(-3, 4), 2))
        value = sp.simplify(F_expr.subs(x, b) - F_expr.subs(x, a))
        if n_terms > 1:
            steps.append(f"將各項不定積分相加：F(x) = {_fmt(F_expr)}")
        steps.append(f"代入上下限：F({b}) - F({a}) = {_fmt(value)}")
        return {
            "topic": "integral",
            "topic_zh": "積分（定積分）",
            "question": f"求定積分：∫[{a} 到 {b}] {_fmt(f_expr)} dx",
            "steps": steps,
            "answer": f"= {_fmt(value)}",
            "func": f_expr,
            "antideriv": F_expr,
            "definite": True,
            "bounds": (a, b),
            "value": value,
        }

    if n_terms > 1:
        steps.append(f"將各項不定積分相加並加上常數 C：F(x) = {_fmt(F_expr)} + C")
    return {
        "topic": "integral",
        "topic_zh": "積分（不定積分）",
        "question": f"求不定積分：∫ {_fmt(f_expr)} dx",
        "steps": steps,
        "answer": f"= {_fmt(F_expr)} + C",
        "func": f_expr,
        "antideriv": F_expr,
        "definite": False,
        "bounds": None,
        "value": None,
    }


# ---------------------------------------------------------------------------
# Limit
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


def format_problem(problem: dict) -> str:
    lines = [f"【sinco 微積分出題：{problem['topic_zh']}】", "", f"題目：{problem['question']}", "", "詳解："]
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
