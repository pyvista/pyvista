"""Fail when a ``vtk_version_info`` comparison can no longer change its result."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pyvista as pv
from tests.conftest import PYVISTA_ROOT_DIR

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SOURCE_DIRS = ('pyvista', 'tests')
SKIP_PARTS = {'build', 'dist', '_build', '__pycache__', '.git'}

# Constant once the bound is reached.
_INCLUSIVE_OPS = (ast.Lt, ast.GtE)
# Constant only strictly below the bound.
_EXCLUSIVE_OPS = (ast.LtE, ast.Gt, ast.Eq, ast.NotEq)

_MIRRORED = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}


def _iter_source_files() -> Iterator[Path]:
    """Yield every Python file in the scanned source directories."""
    for directory in SOURCE_DIRS:
        for path in sorted((PYVISTA_ROOT_DIR / directory).rglob('*.py')):
            if SKIP_PARTS.isdisjoint(path.parts):
                yield path


def _version_bound(node: ast.expr) -> tuple[int, ...] | None:
    """Return the literal version tuple a node represents, if it is one."""
    if (
        isinstance(node, ast.Tuple)
        and node.elts
        and all(
            isinstance(element, ast.Constant) and isinstance(element.value, int)
            for element in node.elts
        )
    ):
        return tuple(element.value for element in node.elts)  # type: ignore[misc]
    return None


def _reads_vtk_version(node: ast.expr) -> bool:
    """Return True if the node reads ``vtk_version_info``."""
    if isinstance(node, ast.Name):
        return node.id == 'vtk_version_info'
    return isinstance(node, ast.Attribute) and node.attr == 'vtk_version_info'


def _dead_gates(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield the line and source of every constant version comparison in a module."""
    minimum = tuple(pv._MIN_SUPPORTED_VTK_VERSION)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        operator = node.ops[0]
        if _reads_vtk_version(node.left):
            bound = _version_bound(node.comparators[0])
        elif _reads_vtk_version(node.comparators[0]):
            bound = _version_bound(node.left)
            operator = _MIRRORED.get(type(operator), type(operator))()
        else:
            continue
        if bound is None:
            continue
        constant = (isinstance(operator, _INCLUSIVE_OPS) and bound <= minimum) or (
            isinstance(operator, _EXCLUSIVE_OPS) and bound < minimum
        )
        if constant:
            yield node.lineno, ast.unparse(node)


def test_no_dead_vtk_version_gates():
    """Every ``vtk_version_info`` comparison must still be able to go both ways."""
    dead = [
        f'{path.relative_to(PYVISTA_ROOT_DIR)}:{lineno}  {source}'
        for path in _iter_source_files()
        for lineno, source in _dead_gates(ast.parse(path.read_text()))
    ]
    assert not dead, (
        f'VTK {pv._MIN_SUPPORTED_VTK_VERSION} is the minimum supported version, so these '
        'comparisons are constant and the code they guard is dead:\n  ' + '\n  '.join(dead)
    )


def test_dead_vtk_version_gates_detected(monkeypatch):
    """Raising the minimum must flag the gates it makes constant."""
    monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', (9, 5, 0))
    source = (
        'if vtk_version_info >= (9, 5, 0):\n    pass\nif vtk_version_info < (9, 9):\n    pass\n'
    )
    assert list(_dead_gates(ast.parse(source))) == [(1, 'vtk_version_info >= (9, 5, 0)')]
