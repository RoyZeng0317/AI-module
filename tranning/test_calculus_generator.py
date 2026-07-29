"""Tests for calculus_generator.py.

Unlike the neural-model test files in this folder (which can only smoke-test
the pipeline since there's no real training data yet), every problem here is
fully deterministic symbolic math — so these tests re-derive the correct
answer independently with sp.diff/sp.integrate/sp.limit and assert it
matches what the generator claims, not just that nothing crashed.
"""

import pytest
import sympy as sp

from calculus_generator import format_problem, generate_problem, x


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

    # both branches (definite/indefinite) should show up over 30 random seeds
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
    another via sympy's auto-combining Add (see module docstring on why
    that matters: it would leave the step list explaining a term that no
    longer appears in the displayed question).
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
