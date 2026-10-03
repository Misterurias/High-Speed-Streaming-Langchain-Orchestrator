"""
The calculator tool: safely computes expressions like "(3 + 4) * sqrt(16)".

Why not just use Python's eval()?
    The expression is written by the router model, and eval() would run ANY
    Python code it was given, e.g. "__import__('os').system('rm -rf ~')".

How this works instead:
    1. `ast.parse` turns the text into a tree (an "abstract syntax tree"):
           "3 + 4 * 2"   →   Add( 3, Mult(4, 2) )
    2. `_eval` walks the tree and only knows how to handle an allow-list of
       pieces: numbers, + - * / // % **, the constants pi and e, and a short
       list of math functions. Anything else (names, imports, attribute
       access, strings, lists) raises MathError.
    3. Size limits stop inputs that are valid math but would freeze the server,
       like 9**9**9 (a number with hundreds of millions of digits).

This file has no LangChain or LLM code. It's plain, deterministic Python,
which is the point: the model decides WHAT to compute, this computes it.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Callable

MAX_EXPR_CHARS = 200
MAX_INT_BITS = 10_000  # ~3,000 digits; stops 9**9**9-style CPU/memory bombs
MAX_FACTORIAL = 500

_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
}
_CONSTS = {"pi": math.pi, "e": math.e}

Number = int | float


class MathError(ValueError):
    """Raised when an expression is invalid, unsafe, or can't be computed."""


def evaluate(expression: str) -> Number:
    """Safely evaluate an arithmetic expression like '(3 + 4) * sqrt(16)'."""
    expr = expression.strip().replace("^", "**")
    if not expr:
        raise MathError("empty expression")
    if len(expr) > MAX_EXPR_CHARS:
        raise MathError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise MathError("not a valid arithmetic expression") from exc
    return _eval(tree.body)


def _check(value: Number) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MathError("result is not a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise MathError("result is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise MathError("result is not finite")
    return value


def _eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathError("only numbers are allowed")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise MathError(f"unknown name '{node.id}'")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check(_UNARY_OPS[type(node.op)](_eval(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # Estimate the size of the result before computing it.
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if left.bit_length() * right > MAX_INT_BITS:
                    raise MathError("result is too large")
            if abs(right) > MAX_INT_BITS:
                raise MathError("exponent is too large")
        try:
            return _check(_BIN_OPS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise MathError("division by zero") from exc
        except OverflowError as exc:
            raise MathError("result is too large") from exc

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        args = [_eval(arg) for arg in node.args]
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise MathError("factorial argument is too large")
        try:
            return _check(_FUNCS[node.func.id](*args))
        except (ValueError, TypeError, OverflowError) as exc:
            raise MathError(f"{node.func.id}: {exc}") from exc

    raise MathError(f"unsupported syntax: {type(node).__name__}")


def is_bare_expression(text: str) -> bool:
    """True if the text is itself an arithmetic expression (e.g. '12*(3+4)').

    Used for the fast path: these queries skip the LLM router entirely.
    A lone number like '42' doesn't count, since there's nothing to compute.
    """
    candidate = text.strip().rstrip("=?").strip()
    if not candidate or len(candidate) > MAX_EXPR_CHARS:
        return False
    try:
        tree = ast.parse(candidate.replace("^", "**"), mode="eval")
    except SyntaxError:
        return False
    if isinstance(tree.body, ast.Constant):
        return False
    try:
        _eval(tree.body)
    except MathError:
        return False
    return True


def format_result(value: Number) -> str:
    """Readable output: 7.0 -> '7', 0.1 + 0.2 -> '0.3'."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)
