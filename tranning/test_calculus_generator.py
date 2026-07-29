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
    explain_derivative,
    explain_integral,
    explain_limit,
    format_problem,
    format_question,
    generate_problem,
    x,
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


def test_explain_derivative_falls_back_generically_for_a_product_of_two_factors():
    expr = x * sp.sin(x)  # not one of the three known single-term patterns
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
