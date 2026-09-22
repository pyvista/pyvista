"""Test that the documentation build reports broken links to documented type aliases."""

from __future__ import annotations

import ast
import re

import pytest

from tests.conftest import PYVISTA_ROOT_DIR

CONF_PY = PYVISTA_ROOT_DIR / 'doc' / 'source' / 'conf.py'


def conf_value(name: str) -> object:
    """Return the literal assigned to ``name`` in the documentation ``conf.py``."""
    for node in ast.parse(CONF_PY.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    msg = f'{name} is not assigned a literal in {CONF_PY}'
    raise KeyError(msg)


@pytest.mark.parametrize('alias', conf_value('_TYPE_ALIASES'))
@pytest.mark.parametrize('prefix', ['', 'pyvista.'])
def test_type_alias_links_are_checked(alias, prefix):
    """Confirm no nitpick ignore hides an unresolved link to a documented type alias."""
    target = f'{prefix}{alias}'
    patterns = [
        pattern
        for _, pattern in conf_value('nitpick_ignore_regex')
        if re.fullmatch(pattern, target)
    ]
    assert not patterns, f'{patterns} hide broken links to {target}'
