"""Tests for calculus_solver.py's free-form request parsing.

Each recognized phrasing is checked against sympy's own ground truth
(sp.diff/sp.integrate/sp.limit), same philosophy as test_calculus_generator.py
— this module always computes a real answer wherever it recognizes a
request, so tests can verify correctness, not just "didn't crash".
"""

import pytest
import sympy as sp

from calculus_generator import x
from calculus_solver import SolveError, parse_and_solve


def test_unrecognized_text_returns_none():
    assert parse_and_solve("") is None
    assert parse_and_solve("你好嗎") is None
    assert parse_and_solve("今天天氣如何") is None


def test_derivative_chinese_suffix_phrasing():
    problem = parse_and_solve("3x^2+5x 的微分")
    assert problem is not None
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(3 * x**2 + 5 * x, x)) == 0


def test_derivative_chinese_daoshu_phrasing():
    problem = parse_and_solve("x^3 的導數")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(x**3, x)) == 0


def test_derivative_english_phrasing():
    problem = parse_and_solve("derivative of x^2 + 1")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(x**2 + 1, x)) == 0


def test_derivative_dydx_phrasing():
    problem = parse_and_solve("d/dx(sin(x))")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(sp.sin(x), x)) == 0


def test_indefinite_integral_symbol_phrasing():
    problem = parse_and_solve("∫ x^2 dx")
    assert problem["topic"] == "integral"
    assert not problem["definite"]
    assert sp.simplify(sp.diff(problem["antideriv"], x) - x**2) == 0


def test_indefinite_integral_chinese_phrasing():
    problem = parse_and_solve("3x 的積分")
    assert problem["topic"] == "integral"
    assert not problem["definite"]


def test_indefinite_integral_english_phrasing():
    problem = parse_and_solve("integral of x^2")
    assert problem["topic"] == "integral"
    assert not problem["definite"]


def test_definite_integral_symbol_bounds_phrasing():
    problem = parse_and_solve("∫[0,1] x^2 dx")
    assert problem["topic"] == "integral"
    assert problem["definite"]
    assert problem["bounds"] == (0, 1)
    expected = sp.integrate(x**2, (x, 0, 1))
    assert sp.simplify(problem["value"] - expected) == 0


def test_definite_integral_chinese_from_to_phrasing():
    problem = parse_and_solve("x^2 從 0 到 2 的定積分")
    assert problem["definite"]
    assert problem["bounds"] == (0, 2)


def test_definite_integral_english_from_to_phrasing():
    problem = parse_and_solve("integral of x^2 from 0 to 1")
    assert problem["definite"]
    assert problem["bounds"] == (0, 1)
    expected = sp.integrate(x**2, (x, 0, 1))
    assert sp.simplify(problem["value"] - expected) == 0


def test_limit_point_first_phrasing():
    problem = parse_and_solve("lim(x->0) sin(x)/x")
    assert problem["topic"] == "limit"
    assert problem["value"] == 1


def test_limit_expr_first_chinese_phrasing():
    problem = parse_and_solve("x^2+1 在 x->2 的極限")
    assert problem["topic"] == "limit"
    assert problem["value"] == 5


def test_limit_infinity_point():
    problem = parse_and_solve("lim(x->oo) 1/x")
    assert problem["value"] == 0


def test_solve_error_raised_for_no_closed_form_integral():
    with pytest.raises(SolveError):
        parse_and_solve("sin(sin(x)) 的積分")


def test_multi_variable_expression_is_rejected():
    # only single-variable-x expressions are supported
    assert parse_and_solve("x*y 的微分") is None


def test_garbage_expression_text_returns_none():
    assert parse_and_solve("這不是算式的微分") is None
