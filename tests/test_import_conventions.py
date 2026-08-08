"""Guard the stdlib import conventions against drift.

The rule, the two lists, and why they live in separate files are documented
under "Import Conventions" in ``CONTRIBUTING.rst``. These tests make the
relationship between the lists machine-checked rather than a matter of two
comments staying honest: they must be disjoint, and together they must govern
every stdlib module the repository actually imports.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

import pytest

# `tomllib` is Python 3.11+, and this project supports 3.10. The convention
# guard only needs to run on one interpreter to be effective, so skip rather
# than take on a `tomli` dependency for the oldest supported version.
tomllib = pytest.importorskip('tomllib', reason='tomllib requires Python 3.11+')

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).parent.parent
SOURCE_DIRS = ('pyvista', 'tests', 'doc', 'examples')
SKIP_PARTS = {'build', 'dist', '_build', '__pycache__', '.git'}

# Aliased imports are an intentional escape hatch, so the modules only ever
# reached that way are governed by neither list.
WAIVED = {
    'importlib.resources',  # from importlib.resources import files as _resources_files
    'xml.dom.minidom',  # import xml.dom.minidom as md
    'xml.etree',  # from xml.etree import ElementTree as ET
}


def _iter_source_files() -> Iterator[Path]:
    for directory in SOURCE_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob('*.py')):
            if not SKIP_PARTS.isdisjoint(path.parts):
                continue
            yield path


def _namespace_modules() -> set[str]:
    """Modules ruff requires to be namespace-imported."""
    with (REPO_ROOT / 'pyproject.toml').open('rb') as file:
        config = tomllib.load(file)
    conventions = config['tool']['ruff']['lint']['flake8-import-conventions']
    return set(conventions['banned-from'])


def _member_modules() -> set[str]:
    """Modules the pygrep hook requires to be member-imported."""
    text = (REPO_ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
    hook = re.search(
        r'- id: namespace-stdlib-imports\b.*?^\s*entry: \'(?P<entry>.*?)\'$',
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert hook is not None, (
        "Could not find the 'namespace-stdlib-imports' hook in "
        '.pre-commit-config.yaml. If it was renamed or restructured, update '
        'this test to match -- the convention still needs a guard.'
    )
    alternation = re.search(r'import\\s\+\(\?:(?P<names>[^)]+)\)', hook.group('entry'))
    assert alternation is not None, (
        f'Could not parse module names out of the hook pattern: {hook.group("entry")!r}'
    )
    return {name.replace('\\.', '.') for name in alternation.group('names').split('|')}


def _documented_member_modules() -> set[str]:
    """The member list as spelled out in CONTRIBUTING.rst."""
    text = (REPO_ROOT / 'CONTRIBUTING.rst').read_text(encoding='utf-8')
    listing = re.search(
        r'The member list is closed and short:(?P<body>.*?)\.\n\n', text, flags=re.DOTALL
    )
    assert listing is not None, (
        'Could not find the member list in CONTRIBUTING.rst. If the wording '
        'changed, update this test -- the list still needs to match the hook.'
    )
    return set(re.findall(r'``([a-z_]+(?:\.[a-z_]+)?)``', listing.group('body')))


def _imported_stdlib_modules() -> dict[str, set[str]]:
    """Map each imported stdlib module to the files importing it."""
    found: dict[str, set[str]] = {}
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split('.')[0] in sys.stdlib_module_names:
                    found.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def test_convention_lists_are_disjoint():
    """A module cannot be required to use both import forms."""
    overlap = _namespace_modules() & _member_modules()
    assert not overlap, (
        f'These modules appear in both convention lists: {sorted(overlap)}. '
        'A module must be either namespace-imported (ruff banned-from in '
        'pyproject.toml) or member-imported (the namespace-stdlib-imports '
        'hook in .pre-commit-config.yaml), never both.'
    )


def test_documentation_matches_the_enforced_member_list():
    """CONTRIBUTING.rst must list exactly what the hook enforces."""
    documented, enforced = _documented_member_modules(), _member_modules()
    assert documented == enforced, (
        f'CONTRIBUTING.rst and the namespace-stdlib-imports hook disagree. '
        f'Only in the docs: {sorted(documented - enforced)}. '
        f'Only in the hook: {sorted(enforced - documented)}.'
    )


def test_convention_lists_are_not_empty():
    """Guard against a parsing change silently emptying either list."""
    assert len(_namespace_modules()) > 20
    assert len(_member_modules()) > 5


@pytest.mark.parametrize('waived', sorted(WAIVED))
def test_waived_modules_are_not_also_governed(waived):
    """A waived submodule should not also appear in a convention list."""
    assert waived not in _namespace_modules() | _member_modules(), (
        f'{waived} is waived in this test but also listed in a convention '
        'list. Remove it from WAIVED, or from the list.'
    )


def test_every_imported_stdlib_module_is_governed():
    """No stdlib module may be imported without a convention governing it.

    This is the check that earns its keep: it fails when someone starts using
    a module that neither list mentions, instead of letting it settle into
    whichever form the first author happened to pick.
    """
    governed = _namespace_modules() | _member_modules() | WAIVED
    ungoverned: dict[str, set[str]] = {}
    for module, files in _imported_stdlib_modules().items():
        if module in governed or module.split('.')[0] in governed:
            continue
        ungoverned[module] = files

    assert not ungoverned, (
        'These stdlib modules are imported but governed by neither import '
        'convention:\n'
        + '\n'.join(
            f'  {module}: {", ".join(sorted(files)[:3])}'
            for module, files in sorted(ungoverned.items())
        )
        + '\n\nAdd each to the namespace list (ruff banned-from in '
        'pyproject.toml) if it exports actions, or to the member list (the '
        'namespace-stdlib-imports hook in .pre-commit-config.yaml) if it '
        'exports types. See "Import Conventions" in CONTRIBUTING.rst.'
    )
