"""Tests for calculus_generator.py.

Unlike the neural-model test files in this folder (which can only smoke-test
the pipeline since there's no real training data yet), every problem here is
fully deterministic symbolic math — so these tests re-derive the correct
answer independently with sp.diff/sp.integrate/sp.limit and assert it
matches what the generator/explainer claims, not just that nothing crashed.
"""

import pytest
import sympy as sp

from calculus_generator import (
    _DERIVATIVE_FORMULA_LABELS,
    explain_derivative,
    explain_gradient,
    explain_integral,
    explain_limit,
    explain_nth_derivative,
    explain_partial_derivative,
    explain_taylor_series,
    format_problem,
    format_question,
    generate_problem,
    x,
    y,
)


def test_random_topic_smoke_across_seeds():
    for seed in range(10):
        problem = generate_problem(seed=seed)
        assert problem["topic"] in {"derivative", "integral", "limit"}
        assert len(problem["steps"]) >= 1
        text = format_problem(problem)
        assert problem["question"] in text
        assert problem["answer"] in text


def test_topic_none_and_literal_random_both_pick_a_topic():
    assert generate_problem(topic=None, seed=1)["topic"] in {"derivative", "integral", "limit"}
    assert generate_problem(topic="random", seed=1)["topic"] in {"derivative", "integral", "limit"}


def test_same_seed_is_reproducible():
    p1 = generate_problem(topic="derivative", seed=123)
    p2 = generate_problem(topic="derivative", seed=123)
    assert p1["question"] == p2["question"]
    assert p1["answer"] == p2["answer"]
    assert p1["steps"] == p2["steps"]


def test_unknown_topic_raises():
    with pytest.raises(ValueError):
        generate_problem(topic="not-a-real-topic")


def test_derivative_matches_sympy_diff():
    for seed in range(15):
        problem = generate_problem(topic="derivative", seed=seed)
        assert problem["topic"] == "derivative"
        assert sp.simplify(sp.diff(problem["func"], x) - problem["deriv"]) == 0
        assert problem["answer"].startswith("f'(x) =")


def test_integral_antiderivative_matches_sympy_integrate():
    saw_definite = saw_indefinite = False
    for seed in range(30):
        problem = generate_problem(topic="integral", seed=seed)
        assert problem["topic"] == "integral"
        assert sp.simplify(sp.diff(problem["antideriv"], x) - problem["func"]) == 0

        if problem["definite"]:
            saw_definite = True
            a, b = problem["bounds"]
            expected = sp.simplify(problem["antideriv"].subs(x, b) - problem["antideriv"].subs(x, a))
            assert sp.simplify(expected - problem["value"]) == 0
            assert "定積分" in problem["topic_zh"]
        else:
            saw_indefinite = True
            assert problem["value"] is None
            assert problem["bounds"] is None
            assert "不定積分" in problem["topic_zh"]
            assert "+ C" in problem["answer"]

    assert saw_definite and saw_indefinite


def test_limit_value_matches_sympy_limit():
    for seed in range(30):
        problem = generate_problem(topic="limit", seed=seed)
        assert problem["topic"] == "limit"
        recomputed = sp.limit(problem["expr"], x, problem["point"])
        assert recomputed == problem["value"]


def test_derivative_term_count_matches_step_explanations():
    """rng.sample (not rng.choice) picks generators without replacement, so
    the number of term-explanation steps should always equal the number of
    distinct terms actually summed into f(x) — no term silently merges into
    another via sympy's auto-combining Add.
    """
    for seed in range(20):
        problem = generate_problem(topic="derivative", seed=seed)
        steps = problem["steps"]
        explained_terms = len(steps) if len(steps) == 1 else len(steps) - 1
        assert len(sp.Add.make_args(problem["func"])) == explained_terms


def test_format_problem_contains_all_sections():
    problem = generate_problem(topic="derivative", seed=1)
    text = format_problem(problem)
    assert "題目" in text
    assert "詳解" in text
    assert "答案" in text
    assert "sinco 微積分出題" in text


def test_format_question_has_no_steps_or_answer():
    problem = generate_problem(topic="derivative", seed=1)
    text = format_question(problem)
    assert problem["question"] in text
    assert problem["answer"] not in text
    for step in problem["steps"]:
        assert step not in text
    assert "解題" in text  # hint to reveal the solution later


# ---------------------------------------------------------------------------
# explain_*() — solving an arbitrary caller-supplied expression
# ---------------------------------------------------------------------------

def test_explain_derivative_on_hand_written_expression():
    expr = 3 * x**2 + 5 * x
    problem = explain_derivative(expr)
    assert problem["deriv"] == sp.diff(expr, x)
    assert problem["answer"] == "f'(x) = 6x + 5"


def test_explain_derivative_product_rule():
    expr = x * sp.sin(x)
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert "乘積律" in problem["steps"][0]


def test_explain_derivative_quotient_rule():
    expr = x / (x + 1)
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert "商法則" in problem["steps"][0]


def test_explain_derivative_chain_rule_nonlinear_inner():
    expr = sp.sin(x**2 + 1)
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert "鏈鎖法則" in problem["steps"][0]
    assert "u=" in problem["steps"][0]


def test_explain_derivative_chain_rule_linear_inner_wording_unchanged():
    # existing quiz term builders (_term_trig/_term_exp) only ever produce a
    # linear inner (c*x) — that wording must stay exactly as before, only
    # the *nonlinear*-inner case above gets the new generalized phrasing.
    expr = 3 * sp.sin(2 * x)
    problem = explain_derivative(expr)
    assert problem["steps"][0] == "對 3sin(2x) 用鏈鎖法則微分 sin(cx)：(3sin(2x))' = 6cos(2x)"


def test_explain_derivative_falls_back_generically_for_three_factor_product():
    # three x-dependent factors is beyond the two-factor product rule this
    # module attempts — the answer is still exact (sympy), wording falls back
    expr = sp.sin(x) * sp.cos(x) * sp.exp(x)
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert "sympy 計算" in problem["steps"][0]


def test_explain_integral_indefinite_and_definite():
    expr = 2 * x**3
    indefinite = explain_integral(expr)
    assert sp.simplify(sp.diff(indefinite["antideriv"], x) - expr) == 0
    assert "+ C" in indefinite["answer"]

    definite = explain_integral(expr, bounds=(0, 2))
    expected = sp.integrate(expr, (x, 0, 2))
    assert sp.simplify(definite["value"] - expected) == 0


def test_explain_integral_raises_when_sympy_cannot_find_closed_form():
    # sin(sin(x)) has no closed-form antiderivative (not even in terms of
    # sympy's special functions) — sympy leaves an unevaluated Integral,
    # which must surface as an error, not a bogus answer.
    with pytest.raises(ValueError):
        explain_integral(sp.sin(sp.sin(x)))


def test_explain_limit_direct_substitution():
    problem = explain_limit(x**2 + 1, 2)
    assert problem["value"] == 5
    assert "有定義" in problem["steps"][0]


def test_explain_limit_factor_and_cancel_zero_over_zero():
    expr = (x**2 - 4) / (x - 2)
    problem = explain_limit(expr, 2)
    assert problem["value"] == 4
    assert "0/0" in problem["steps"][0]
    assert "約去" in problem["steps"][1]


def test_explain_limit_generic_fallback_for_trig_zero_over_zero():
    expr = sp.sin(x) / x
    problem = explain_limit(expr, 0)
    assert problem["value"] == 1
    assert "sympy 直接計算" in problem["steps"][0]


def test_explain_limit_at_infinity():
    expr = (3 * x**2 + 1) / (x**2 - 5)
    problem = explain_limit(expr, sp.oo)
    assert problem["value"] == 3
    assert "∞" in problem["question"]


def test_explain_integral_u_substitution():
    expr = x * sp.sin(x**2)
    problem = explain_integral(expr)
    assert sp.simplify(sp.diff(problem["antideriv"], x) - expr) == 0
    assert "湊微分" in problem["steps"][0]


def test_explain_integral_u_substitution_declines_when_ratio_still_has_x():
    # x*sin(x) looks superficially similar to x*sin(x**2) but isn't a valid
    # u-substitution (needs integration by parts instead) — must fall back,
    # not produce a bogus "u-substitution" step with the wrong math.
    expr = x * sp.sin(x)
    problem = explain_integral(expr)
    assert sp.simplify(sp.diff(problem["antideriv"], x) - expr) == 0
    assert "湊微分" not in problem["steps"][0]


def test_quiz_derivative_term_builders_include_product_and_chain():
    seen_product = seen_chain = False
    for seed in range(60):
        problem = generate_problem(topic="derivative", seed=seed)
        for step in problem["steps"]:
            if "乘積律" in step:
                seen_product = True
            if "鏈鎖法則，令 u=" in step:
                seen_chain = True
    assert seen_product and seen_chain


def test_quiz_integral_term_builder_includes_u_substitution():
    seen = False
    for seed in range(60):
        problem = generate_problem(topic="integral", seed=seed)
        for step in problem["steps"]:
            if "湊微分" in step:
                seen = True
    assert seen


# ---------------------------------------------------------------------------
# Phase 2: more function types (log/sqrt/反三角/雙曲)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr,formula_hint",
    [
        (sp.log(x), "(ln x)'"),
        (sp.sqrt(x), "(√x)'"),
        (sp.asin(x), "(arcsin x)'"),
        (sp.acos(x), "(arccos x)'"),
        (sp.atan(x), "(arctan x)'"),
        (sp.sinh(2 * x), "(sinh x)'"),
        (sp.cosh(3 * x), "(cosh x)'"),
        (sp.tanh(x), "(tanh x)'"),
    ],
)
def test_explain_derivative_more_function_types_linear_inner(expr, formula_hint):
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert formula_hint in problem["steps"][0]


def test_explain_derivative_more_function_type_nonlinear_inner_uses_chain_wording():
    expr = sp.log(x**2 + 1)
    problem = explain_derivative(expr)
    assert sp.simplify(problem["deriv"] - sp.diff(expr, x)) == 0
    assert "鏈鎖法則，令 u=" in problem["steps"][0]
    assert "ln(u)" in problem["steps"][0]  # display name, not sympy's raw "log"


def test_fmt_renames_sympy_function_names_to_chinese_textbook_spelling():
    problem = explain_derivative(sp.sqrt(x))
    assert "sqrt(" not in problem["question"]
    assert "√(" in problem["question"]
    problem = explain_derivative(sp.log(x))
    assert "log(" not in problem["question"]
    assert "ln(" in problem["question"]


def test_explain_integral_sqrt_formula():
    problem = explain_integral(sp.sqrt(x))
    assert sp.simplify(sp.diff(problem["antideriv"], x) - sp.sqrt(x)) == 0
    assert "√x dx" in problem["steps"][0]


def test_explain_integral_sinh_cosh_formula():
    for expr, hint in [(sp.sinh(2 * x), "sinh x dx"), (sp.cosh(3 * x), "cosh x dx")]:
        problem = explain_integral(expr)
        assert sp.simplify(sp.diff(problem["antideriv"], x) - expr) == 0
        assert hint in problem["steps"][0]


def test_explain_integral_sqrt_u_substitution():
    expr = x * sp.sqrt(x**2 + 1)
    problem = explain_integral(expr)
    assert sp.simplify(sp.diff(problem["antideriv"], x) - expr) == 0
    assert "湊微分" in problem["steps"][0]


def test_explain_derivative_log_asin_still_fall_back_for_integral():
    # log/asin/acos/atan/tanh's antiderivatives need integration by parts,
    # not a clean substitution — this module intentionally doesn't claim a
    # named formula for those; the answer is still exact (sympy), wording
    # just falls back to the generic line, same as any other unrecognized
    # integral shape.
    problem = explain_integral(sp.log(x))
    assert sp.simplify(sp.diff(problem["antideriv"], x) - sp.log(x)) == 0
    assert "sympy 計算" in problem["steps"][0]


def test_quiz_derivative_term_builders_include_more_function_types():
    seen = False
    for seed in range(60):
        problem = generate_problem(topic="derivative", seed=seed)
        for step in problem["steps"]:
            if any(label in step for label in _DERIVATIVE_FORMULA_LABELS.values()):
                seen = True
    assert seen


def test_quiz_integral_term_builder_includes_more_function_types():
    seen = False
    for seed in range(60):
        problem = generate_problem(topic="integral", seed=seed)
        for step in problem["steps"]:
            if "√x dx" in step or "sinh x dx" in step or "cosh x dx" in step:
                seen = True
    assert seen


# ---------------------------------------------------------------------------
# Phase 3: higher-order derivatives + Taylor/Maclaurin series
# ---------------------------------------------------------------------------

def test_explain_nth_derivative_matches_repeated_sympy_diff():
    expr = sp.sin(x)
    for n in range(1, 5):
        problem = explain_nth_derivative(expr, n)
        assert sp.simplify(problem["deriv"] - sp.diff(expr, x, n)) == 0
        assert problem["order"] == n
        assert f"f^({n})(x)" in problem["answer"]


def test_explain_nth_derivative_step_count_scales_with_order():
    # each order reuses explain_derivative()'s own step list in full — a
    # multi-term function's 2nd derivative should have more step lines than
    # its 1st, not a fixed "1 line per order" summary.
    expr = x**4 + 3 * x**2
    p1 = explain_nth_derivative(expr, 1)
    p2 = explain_nth_derivative(expr, 2)
    assert len(p2["steps"]) > len(p1["steps"])
    assert all(step.startswith("[第 1 階]") for step in p1["steps"])
    assert any(step.startswith("[第 2 階]") for step in p2["steps"])


def test_explain_nth_derivative_rejects_non_positive_order():
    with pytest.raises(ValueError):
        explain_nth_derivative(x**2, 0)
    with pytest.raises(ValueError):
        explain_nth_derivative(x**2, -1)


def test_explain_taylor_series_matches_sympy_series():
    for expr, point, order in [
        (sp.exp(x), 0, 4),
        (sp.sin(x), 0, 5),
        (sp.log(x), 1, 3),
        (x**3 - 2 * x, 0, 3),
    ]:
        problem = explain_taylor_series(expr, point, order)
        expected = sp.series(expr, x, point, order + 1).removeO()
        assert sp.expand(problem["series"] - expected) == 0
        assert len(problem["steps"]) == order + 1


def test_explain_taylor_series_point_zero_is_labeled_maclaurin():
    problem = explain_taylor_series(sp.exp(x), 0, 2)
    assert problem["topic_zh"] == "馬克勞林級數"
    problem = explain_taylor_series(sp.exp(x), 1, 2)
    assert problem["topic_zh"] == "泰勒級數"


def test_explain_taylor_series_rejects_negative_order():
    with pytest.raises(ValueError):
        explain_taylor_series(sp.sin(x), 0, -1)


# ---------------------------------------------------------------------------
# Phase 4: multivariable calculus (partial derivatives / gradient)
# ---------------------------------------------------------------------------

def test_explain_partial_derivative_matches_sympy_diff():
    expr = x**2 * y + y**3
    px = explain_partial_derivative(expr, x)
    assert px["deriv"] == sp.diff(expr, x)
    py = explain_partial_derivative(expr, y)
    assert py["deriv"] == sp.diff(expr, y)


def test_explain_partial_derivative_holds_other_var_constant():
    px = explain_partial_derivative(x**2 * y, x)
    assert "y" in px["steps"][0] and "常數" in px["steps"][0]
    py = explain_partial_derivative(x**2 * y, y)
    assert "x" in py["steps"][0] and "常數" in py["steps"][0]


def test_explain_partial_derivative_rejects_wrong_variable():
    with pytest.raises(ValueError):
        explain_partial_derivative(x**2, sp.symbols("z"))


def test_explain_partial_derivative_rejects_extra_free_symbols():
    z = sp.symbols("z")
    with pytest.raises(ValueError):
        explain_partial_derivative(x * z, x)


def test_explain_gradient_matches_sympy_diff():
    expr = sp.sin(x * y) + x**3
    grad = explain_gradient(expr)
    assert grad["gradient"] == (sp.diff(expr, x), sp.diff(expr, y))
    assert "∇f" in grad["answer"]


def test_explain_gradient_single_variable_expression_still_works():
    # a purely-x expression is still a valid (degenerate) 2-variable function
    grad = explain_gradient(x**2)
    assert grad["gradient"] == (2 * x, 0)
