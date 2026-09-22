"""Require docstrings where ruff's D1 rules stop at a private name.

Ruff treats a private class, a private module, and everything inside them as
private, so neither they nor their public members are checked. This applies the
D101/D102/D103 standard there.
"""

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


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
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

    def visit(node: ast.AST, *, inside: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                private = child.name.startswith('_')
                if (private or inside) and not _has_docstring(child):
                    why = 'is private' if private else f'is in {inside}'
                    errors.append(
                        f'{path}:{child.lineno}: class {child.name!r} {why}, '
                        f'so it needs a docstring'
                    )
                visit(
                    child, inside=inside or f'private class {child.name!r}' if private else inside
                )
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if inside and not child.name.startswith('_'):
                    decorators = _decorator_names(child)
                    exempt = {'overload', 'setter', 'deleter'} & set(decorators)
                    if not exempt and not _has_docstring(child):
                        errors.append(
                            f'{path}:{child.lineno}: public {child.name!r} is in {inside}, '
                            f'so it needs a docstring'
                        )

    # Ruff skips a private module wholesale, so its public defs are unchecked too
    private_module = any(part.startswith('_') for part in path.with_suffix('').parts)
    visit(
        ast.parse(path.read_text(encoding='utf-8')),
        inside='a private module' if private_module else None,
    )
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
    raise SystemExit(main(sys.argv[1:]))
