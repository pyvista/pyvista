"""Guard the stdlib import conventions against drift.

Two complementary lists govern how stdlib modules may be imported:

* ``banned-from`` under ``[tool.ruff.lint.flake8-import-conventions]`` in
  ``pyproject.toml`` -- modules that must be namespace-imported
  (``re.escape``, ``functools.wraps``).
* the ``namespace-stdlib-imports`` pygrep hook in ``.pre-commit-config.yaml``
  -- modules that must be member-imported (``from pathlib import Path``).

They live in separate files because ruff can only express one direction. These
tests make the relationship between them machine-checked rather than a matter
of two comments staying honest: the lists must be disjoint, and together they
must govern every stdlib module the repository actually imports. A module
belonging to neither list fails here rather than drifting silently.
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

# Submodule idioms that neither list governs. `banned-from` does not match
# submodules, so banning the parent leaves these untouched -- which is what we
# want, since the submodule must be imported explicitly to be usable.
WAIVED = {
    'http.server',
    'importlib.metadata',
    'importlib.resources',
    'importlib.util',
    'xml.dom.minidom',
    'xml.etree',
    'xml.etree.ElementTree',
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
        'exports types used in annotation or base-class positions.'
    )
