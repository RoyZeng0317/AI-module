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


def test_derivative_spoken_suffix_with_trailing_question_mark():
    problem = parse_and_solve("x^2+1, x 微分後是多少?")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(x**2 + 1, x)) == 0


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


def test_derivative_product_rule_phrasing():
    problem = parse_and_solve("x*sin(x) 的微分")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(x * sp.sin(x), x)) == 0
    assert "乘積律" in problem["steps"][0]


def test_derivative_chain_rule_nonlinear_inner_phrasing():
    problem = parse_and_solve("sin(x^2) 的微分")
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(sp.sin(x**2), x)) == 0
    assert "鏈鎖法則" in problem["steps"][0]


@pytest.mark.parametrize(
    "text,expected_expr",
    [
        ("sqrt(x) 的微分", sp.sqrt(x)),
        ("arcsin(x) 的微分", sp.asin(x)),
        ("asin(x) 的微分", sp.asin(x)),
        ("arccos(x) 的微分", sp.acos(x)),
        ("arctan(x) 的微分", sp.atan(x)),
        ("sinh(2x) 的微分", sp.sinh(2 * x)),
        ("cosh(3x) 的微分", sp.cosh(3 * x)),
        ("tanh(x) 的微分", sp.tanh(x)),
    ],
)
def test_derivative_more_function_types_phrasing(text, expected_expr):
    problem = parse_and_solve(text)
    assert problem["topic"] == "derivative"
    assert sp.simplify(problem["deriv"] - sp.diff(expected_expr, x)) == 0


def test_integral_sqrt_phrasing():
    problem = parse_and_solve("sqrt(x) 的積分")
    assert problem["topic"] == "integral"
    assert sp.simplify(sp.diff(problem["antideriv"], x) - sp.sqrt(x)) == 0


def test_nth_derivative_chinese_ordinal_phrasing():
    problem = parse_and_solve("sin(x) 的三階導數")
    assert problem["topic"] == "nth_derivative"
    assert problem["order"] == 3
    assert sp.simplify(problem["deriv"] - sp.diff(sp.sin(x), x, 3)) == 0


def test_nth_derivative_arabic_digit_phrasing():
    problem = parse_and_solve("x^4 的2階導數")
    assert problem["order"] == 2
    assert sp.simplify(problem["deriv"] - sp.diff(x**4, x, 2)) == 0


def test_nth_derivative_ddx_notation_phrasing():
    problem = parse_and_solve("d^2/dx^2(x^3+2x)")
    assert problem["order"] == 2
    assert sp.simplify(problem["deriv"] - sp.diff(x**3 + 2 * x, x, 2)) == 0


def test_taylor_series_phrasing_at_nonzero_point():
    problem = parse_and_solve("ln(x) 在 x=1 展開到第3階泰勒級數")
    assert problem["topic"] == "taylor_series"
    assert problem["point"] == 1
    assert problem["order"] == 3
    expected = sp.series(sp.log(x), x, 1, 4).removeO()
    assert sp.expand(problem["series"] - expected) == 0


def test_maclaurin_phrasing():
    problem = parse_and_solve("sin(x) 的5階馬克勞林展開")
    assert problem["topic"] == "taylor_series"
    assert problem["point"] == 0
    assert problem["order"] == 5
    expected = sp.series(sp.sin(x), x, 0, 6).removeO()
    assert sp.expand(problem["series"] - expected) == 0


def test_partial_derivative_chinese_phrasing():
    from calculus_generator import y

    problem = parse_and_solve("x^2*y+y^3 對 x 的偏微分")
    assert problem["topic"] == "partial_derivative"
    assert problem["var"] == x
    assert problem["deriv"] == sp.diff(x**2 * y + y**3, x)


def test_partial_derivative_notation_phrasing():
    from calculus_generator import y

    problem = parse_and_solve("∂/∂y(x^2*y)")
    assert problem["topic"] == "partial_derivative"
    assert problem["var"] == y
    assert problem["deriv"] == sp.diff(x**2 * y, y)


def test_ddx_notation_still_routes_to_single_variable_derivative():
    # regression guard: "d/dx" must NOT be swallowed by the new "∂/∂x"
    # Phase 4 pattern — it stays on the original single-variable path.
    problem = parse_and_solve("d/dx(sin(x))")
    assert problem["topic"] == "derivative"


def test_gradient_phrasing():
    from calculus_generator import y

    problem = parse_and_solve("x^2*y+y^3 的梯度")
    assert problem["topic"] == "gradient"
    expr = x**2 * y + y**3
    assert problem["gradient"] == (sp.diff(expr, x), sp.diff(expr, y))
