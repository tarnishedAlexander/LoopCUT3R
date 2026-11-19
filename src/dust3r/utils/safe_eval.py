import ast
from typing import Any, Mapping


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains unsupported or unsafe syntax."""


_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.MatMult,
)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate_node(node: ast.AST, allowed_names: set[str], context: str) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body, allowed_names, context)
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError(
                f"Unsafe {context} expression: function calls must target named symbols"
            )
        if node.func.id not in allowed_names:
            raise UnsafeExpressionError(
                f"Unknown symbol '{node.func.id}' in {context} expression"
            )
        for arg in node.args:
            _validate_node(arg, allowed_names, context)
        for kw in node.keywords:
            if kw.arg is None:
                raise UnsafeExpressionError(
                    f"Unsafe {context} expression: **kwargs are not allowed"
                )
            _validate_node(kw.value, allowed_names, context)
        return

    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise UnsafeExpressionError(
                f"Unknown symbol '{node.id}' in {context} expression"
            )
        return

    if isinstance(node, ast.Constant):
        return

    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            _validate_node(elt, allowed_names, context)
        return

    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is not None:
                _validate_node(key, allowed_names, context)
            _validate_node(value, allowed_names, context)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError(
                f"Unsafe {context} expression: unsupported binary operator"
            )
        _validate_node(node.left, allowed_names, context)
        _validate_node(node.right, allowed_names, context)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError(
                f"Unsafe {context} expression: unsupported unary operator"
            )
        _validate_node(node.operand, allowed_names, context)
        return

    raise UnsafeExpressionError(
        f"Unsafe {context} expression: unsupported syntax node {type(node).__name__}"
    )


def safe_eval_expr(
    expression: str,
    namespace: Mapping[str, Any],
    *,
    context: str = "expression",
) -> Any:
    """Safely evaluate constructor-style config expressions.

    Supported grammar is intentionally small: literals, tuples/lists/dicts,
    constructor calls to allowed symbols, and basic arithmetic composition.
    """
    if not isinstance(expression, str):
        raise TypeError(f"Expected {context} expression as string, got {type(expression)}")

    parsed = ast.parse(expression, mode="eval")
    allowed_names = set(namespace.keys())
    _validate_node(parsed, allowed_names, context)

    safe_globals = {"__builtins__": {}}
    safe_globals.update(namespace)
    return eval(compile(parsed, filename=f"<{context}>", mode="eval"), safe_globals, {})
