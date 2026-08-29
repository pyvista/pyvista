"""Require docstrings on public members of private classes, which ruff D1 skips."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the rightmost name of each decorator on ``node``."""
    names = []
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether ``node`` starts with a docstring."""
    return (
        bool(node.body)
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def _check_file(path: Path) -> list[str]:
    """Return one error line per undocumented public member of a private class."""
    errors = []

    def visit(node: ast.AST, *, in_private_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, in_private_class=in_private_class or child.name.startswith('_'))
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if in_private_class and not child.name.startswith('_'):
                    decorators = _decorator_names(child)
                    exempt = {'overload', 'setter', 'deleter'} & set(decorators)
                    if not exempt and not _has_docstring(child):
                        errors.append(
                            f'{path}:{child.lineno}: public member '
                            f'{child.name!r} of a private class has no docstring'
                        )

    visit(ast.parse(path.read_text()), in_private_class=False)
    return errors


def main(argv: list[str]) -> int:
    """Check every path given on the command line."""
    all_errors = []
    for filename in argv:
        all_errors.extend(_check_file(Path(filename)))
    for error in all_errors:
        print(error)  # noqa: T201
    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
