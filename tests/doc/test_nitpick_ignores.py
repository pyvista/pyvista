"""Test that the documentation build reports broken links to documented type aliases."""

from __future__ import annotations

import ast

import pytest

from tests.conftest import PYVISTA_ROOT_DIR

CONF_PY = PYVISTA_ROOT_DIR / 'doc' / 'source' / 'conf.py'


def conf_value(name: str) -> object:
    """Return the literal assigned to ``name`` in the documentation ``conf.py``."""
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(CONF_PY.read_text(encoding='utf-8')).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


@pytest.mark.parametrize('name', conf_value('_DOCUMENTED_TYPES'))
def test_documented_types_are_not_ignored(name):
    """Confirm no nitpick ignore hides an unresolved link to a documented type."""
    assert name not in conf_value('_UNDOCUMENTED_TYPES')


def test_undocumented_types_are_sorted():
    """Keep the nitpick ignore names sorted and unique."""
    names = conf_value('_UNDOCUMENTED_TYPES')
    assert names == sorted(set(names))
