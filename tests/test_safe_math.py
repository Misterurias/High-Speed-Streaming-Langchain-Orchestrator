"""Tests for app/safe_math.py, the calculator tool."""

import pytest

from app.safe_math import MathError, evaluate, format_result, is_bare_expression


class TestCalculatesCorrectly:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("2 + 2", 4),
            ("(3 + 4) * 5", 35),
            ("2^10", 1024),             # ^ is accepted as a power sign
            ("10 / 4", 2.5),
            ("-5 + 3", -2),
            ("7 // 2", 3),
            ("7 % 3", 1),
            ("sqrt(144) + abs(-3)", 15),
            ("factorial(5)", 120),
            ("round(pi, 2)", 3.14),
            ("log10(1000)", 3),
        ],
    )
    def test_evaluates_expression(self, expression, expected):
        assert evaluate(expression) == expected


class TestRejectsUnsafeOrInvalidInput:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",
            "open('/etc/passwd')",
            "(1).__class__",
        ],
        ids=["import_os", "open_file", "attribute_access"],
    )
    def test_blocks_code_injection(self, expression):
        with pytest.raises(MathError):
            evaluate(expression)

    @pytest.mark.parametrize(
        "expression",
        ["9**9**9", "2 ** 100000", "factorial(100000)", "10.0 ** 400",
         "1.0001 ** 20000", "10**2000 * 10**2000", "1e308 * 10"],
        ids=["power_tower", "huge_power", "huge_factorial", "float_overflow",
             "huge_float_exponent", "huge_product", "float_infinity"],
    )
    def test_blocks_results_too_large_to_compute_safely(self, expression):
        with pytest.raises(MathError):
            evaluate(expression)

    @pytest.mark.parametrize(
        "expression",
        ["", "1 +", "x + 1", "'a' * 3", "[1, 2]", "1 < 2", "sqrt(x=4)", "True + 1", "1" * 201],
        ids=["empty", "incomplete", "unknown_name", "string", "list",
             "comparison", "keyword_argument", "boolean", "too_long"],
    )
    def test_rejects_things_that_are_not_arithmetic(self, expression):
        with pytest.raises(MathError):
            evaluate(expression)

    @pytest.mark.parametrize(
        "expression", ["1 / 0", "sqrt(-1)", "log(0)", "(-8) ** 0.5"],
        ids=["divide_by_zero", "sqrt_negative", "log_zero", "complex_result"],
    )
    def test_reports_math_errors_instead_of_crashing(self, expression):
        with pytest.raises(MathError):
            evaluate(expression)


class TestDetectsBareExpressions:
    """Bare expressions skip the router LLM entirely (the "fast path")."""

    @pytest.mark.parametrize("text", ["12 * (3 + 4)", "2+2=", "sqrt(16)?"])
    def test_recognizes_pure_math(self, text):
        assert is_bare_expression(text)

    @pytest.mark.parametrize(
        "text", ["42", "what is 2 + 2?", "Tell me about Rome", "", "1/0"],
        ids=["lone_number", "question_with_words", "no_math", "empty", "invalid_math"],
    )
    def test_leaves_everything_else_to_the_router(self, text):
        assert not is_bare_expression(text)


class TestFormatsResults:
    @pytest.mark.parametrize(
        "value, text",
        [(7.0, "7"), (0.1 + 0.2, "0.3"), (10 / 3, "3.33333333333"), (12, "12"), (1e20, "1e+20")],
        ids=["whole_float", "float_noise", "repeating", "int", "huge_float"],
    )
    def test_produces_readable_numbers(self, value, text):
        assert format_result(value) == text
