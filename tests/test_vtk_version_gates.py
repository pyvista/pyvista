"""Fail when a ``vtk_version_info`` comparison can no longer change its result."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from tests.conftest import PYVISTA_ROOT_DIR
from tests.conftest import source_files

if TYPE_CHECKING:
    from collections.abc import Iterator

SOURCE_DIRS = ('pyvista', 'tests')

_DEAD_AT_BOUND = (ast.Lt, ast.GtE)
_DEAD_BELOW_BOUND = (ast.LtE, ast.Gt, ast.Eq, ast.NotEq)

_MIRRORED = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}

_OPERATORS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


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
        return tuple(element.value for element in node.elts)
    return None


def _reads_vtk_version(node: ast.expr) -> bool:
    """Return True if the node reads ``vtk_version_info``."""
    if isinstance(node, ast.Name):
        return node.id == 'vtk_version_info'
    return isinstance(node, ast.Attribute) and node.attr == 'vtk_version_info'


def _is_constant(operator_: ast.cmpop, bound: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    """Return True if the comparison holds the same result for every supported version."""
    return (isinstance(operator_, _DEAD_AT_BOUND) and bound <= minimum) or (
        isinstance(operator_, _DEAD_BELOW_BOUND) and bound < minimum
    )


def _dead_gates(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield the line and source of every constant version comparison in a module."""
    minimum = tuple(pv._MIN_SUPPORTED_VTK_VERSION)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for index, node_op in enumerate(node.ops):
            left, right = operands[index], operands[index + 1]
            operator_ = node_op
            if _reads_vtk_version(left):
                bound = _version_bound(right)
            elif _reads_vtk_version(right):
                bound = _version_bound(left)
                operator_ = _MIRRORED.get(type(node_op), type(node_op))()
            else:
                continue
            if bound is not None and _is_constant(operator_, bound, minimum):
                yield node.lineno, ast.unparse(node)
                break


def test_no_dead_vtk_version_gates():
    """Every ``vtk_version_info`` comparison must still be able to go both ways."""
    dead = [
        f'{path.relative_to(PYVISTA_ROOT_DIR)}:{lineno}  {source}'
        for path in source_files(*SOURCE_DIRS)
        for lineno, source in _dead_gates(ast.parse(path.read_text(encoding='utf-8')))
    ]
    assert not dead, (
        f'VTK {pv._MIN_SUPPORTED_VTK_VERSION} is the minimum supported version, so these '
        'comparisons are constant, so one of the branches they guard is dead:\n  '
        + '\n  '.join(dead)
    )


_MINIMUM = (9, 5, 0)
_BOUNDS = [(9, 4, 0), (9, 5, 0), (9, 5, 2), (9, 6, 0), (9, 5), (9,)]


def _reachable_versions(bound: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Return allowed versions on both sides of a bound, including the bound itself."""
    near = (*bound, 0, 0)[:3]
    candidates = {
        _MINIMUM,
        near,
        (near[0], near[1], near[2] + 1),
        (near[0], near[1] + 1, 0),
        (near[0] + 1, 0, 0),
    }
    return sorted(version for version in candidates if version >= _MINIMUM)


@pytest.mark.parametrize('bound', _BOUNDS)
@pytest.mark.parametrize('symbol', ['<', '<=', '>', '>=', '==', '!='])
@pytest.mark.parametrize('mirrored', [False, True])
def test_dead_gate_matches_brute_force(monkeypatch, symbol, bound, mirrored):
    """The classification must agree with evaluating the gate on either side of the bound."""
    monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', _MINIMUM)
    gate = (
        f'{bound} {symbol} vtk_version_info' if mirrored else f'vtk_version_info {symbol} {bound}'
    )
    node = ast.parse(gate).body[0].value
    operator_ = _OPERATORS[type(node.ops[0])]
    results = {
        operator_(bound, version) if mirrored else operator_(version, bound)
        for version in _reachable_versions(bound)
    }
    expected = [(1, gate)] if len(results) == 1 else []
    assert list(_dead_gates(ast.parse(gate))) == expected


@pytest.mark.parametrize(
    'gate',
    [
        '(9, 4, 0) <= vtk_version_info < (9, 9)',
        'lower <= vtk_version_info < (9, 5, 0)',
    ],
    ids=['dead_first_half', 'dead_second_half'],
)
def test_dead_gate_reports_chained_comparison(monkeypatch, gate):
    """A chained comparison is reported when either half has gone constant."""
    monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', _MINIMUM)
    assert list(_dead_gates(ast.parse(gate))) == [(1, gate)]


@pytest.mark.parametrize(
    'gate', ['version_info < (9, 4, 0)', 'vtk_version_info < other', 'value == (9, 4, 0)']
)
def test_dead_gate_ignores_other_comparisons(monkeypatch, gate):
    """Comparisons that do not read ``vtk_version_info`` are left alone."""
    monkeypatch.setattr(pv, '_MIN_SUPPORTED_VTK_VERSION', _MINIMUM)
    assert list(_dead_gates(ast.parse(gate))) == []


def test_source_files_scanned():
    """The scan must actually reach the package and the test suite."""
    scanned = {path.relative_to(PYVISTA_ROOT_DIR).parts[0] for path in source_files(*SOURCE_DIRS)}
    assert scanned == set(SOURCE_DIRS)
