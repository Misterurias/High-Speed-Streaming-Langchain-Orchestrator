import pytest

from app.safe_math import MathError, evaluate, format_result, is_bare_expression


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 2", 4),
        ("(3 + 4) * 5", 35),
        ("2^10", 1024),
        ("sqrt(144) + abs(-3)", 15),
        ("10 / 4", 2.5),
        ("-5 + 3", -2),
        ("factorial(5)", 120),
        ("round(pi, 2)", 3.14),
    ],
)
def test_valid_expressions(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",  # code injection
        "open('/etc/passwd')",
        "x + 1",                          # unknown name
        "9**9**9",                        # exponent bomb
        "factorial(100000)",
        "1 / 0",
        "'a' * 3",
        "[1, 2, 3]",
        "sqrt(-1)",
        "",
        "1 +" ,
    ],
)
def test_rejected_expressions(expr):
    with pytest.raises(MathError):
        evaluate(expr)


def test_bare_expression_detection():
    assert is_bare_expression("12 * (3 + 4)")
    assert is_bare_expression("2+2=")
    assert not is_bare_expression("42")
    assert not is_bare_expression("what is 2 + 2?")
    assert not is_bare_expression("Tell me about Rome")


def test_format_result():
    assert format_result(7.0) == "7"
    assert format_result(0.1 + 0.2) == "0.3"
    assert format_result(10 / 3) == "3.33333333333"
