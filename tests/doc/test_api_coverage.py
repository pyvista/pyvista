"""Validate that every public top-level ``pyvista`` symbol is referenced in the API docs.

A 2026-04 manual audit of the Sphinx reference found 26 undocumented writer
classes, 17 error classes, and 37 utility functions because nothing in CI
caught a public symbol that lacked an ``autosummary`` entry. This test
closes that gap by intersecting the live ``pyvista`` namespace with the
names referenced under ``doc/source/api/``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import re

import pytest

import pyvista as pv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_DOC_ROOT = _REPO_ROOT / 'doc' / 'source' / 'api'

# Directives whose body is an indented block of symbol names.
_BLOCK_DIRECTIVES = frozenset({'autosummary'})

# Directives that name a single target on the same line.
_TARGET_DIRECTIVES = frozenset(
    {
        'autoclass',
        'autodata',
        'autoenum',
        'autoexception',
        'autofunction',
        'automethod',
        'automodule',
        'autonamedtuple',
        'autoproperty',
        'autotypevar',
    }
)

_DIRECTIVE_RE = re.compile(r'^\.\.\s+(auto\w+)::\s*(\S*)')
_IDENTIFIER_RE = re.compile(r'([A-Za-z_][\w\.]*)')

# Public symbols re-exported at the top level that are intentionally kept out of
# the user-facing reference. Each entry is a low-level VTK/NumPy helper, a
# decorator, or an internal base class that ships public for back-compat but is
# not meant to appear in the end-user Sphinx reference.
_ALLOWED_UNDOCUMENTED = frozenset(
    {
        'AnnotatedIntEnum',  # base class for internal int-enum types
        'Grid',  # abstract base; concrete Grid subclasses are documented
        'PointGrid',  # abstract base; concrete subclasses are documented
        'VersionInfo',  # internal version namedtuple
        'abstract_class',  # decorator, internal
        'assert_empty_kwargs',  # internal kwargs guard
        'check_math_text_support',  # internal matplotlib/vtk compatibility check
        'check_matplotlib_vtk_compatibility',  # internal compatibility check
        'check_valid_vector',  # internal assertion
        'conditional_decorator',  # internal decorator factory
        'convert_string_array',  # low-level VTK string conversion
        'create_mixed_cells',  # low-level cell-array builder
        'get_mixed_cells',  # low-level cell-array reader
        'get_vtk_type',  # low-level VTK dtype helper
        'ncells_from_cells',  # low-level cell-array helper
        'numpy_to_idarr',  # low-level VTK id-array helper
        'parse_field_choice',  # internal association parser
        'raise_has_duplicates',  # internal assertion
        'raise_not_matching',  # internal assertion
        'row_array',  # low-level array-by-row accessor
        'threaded',  # internal threading decorator
        'try_callback',  # internal callback guard
        'vtk_bit_array_to_char',  # low-level VTK bit-array conversion
        'vtk_id_list_to_array',  # low-level VTK id-list conversion
    }
)


def _is_pyvista_owned(obj: object) -> bool:
    module = getattr(obj, '__module__', None)
    return bool(module and module.startswith('pyvista'))


def _collect_public_symbols() -> set[str]:
    """Every class/function on ``pyvista`` that is defined inside ``pyvista``."""
    symbols: set[str] = set()
    for name in dir(pv):
        if name.startswith('_'):
            continue
        obj = getattr(pv, name, None)
        if obj is None:
            continue
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if not _is_pyvista_owned(obj):
            continue
        symbols.add(name)
    return symbols


def _collect_documented_names() -> set[str]:
    """Every identifier referenced under an autodoc directive in ``doc/source/api``.

    Names are stored by their final dotted segment so that an entry like
    ``pyvista.core.errors.MissingDataError`` matches the top-level
    ``pyvista.MissingDataError``.
    """
    names: set[str] = set()
    for rst in _API_DOC_ROOT.rglob('*.rst'):
        lines = rst.read_text(encoding='utf-8').splitlines()
        in_block = False
        block_indent = -1
        for raw in lines:
            stripped = raw.strip()
            match = _DIRECTIVE_RE.match(stripped)
            if match:
                directive, target = match.group(1), match.group(2)
                if directive in _TARGET_DIRECTIVES and target:
                    names.add(target.rsplit('.', 1)[-1])
                if directive in _BLOCK_DIRECTIVES:
                    in_block = True
                    block_indent = len(raw) - len(raw.lstrip())
                else:
                    in_block = False
                continue
            if not in_block:
                continue
            if not stripped:
                continue
            indent = len(raw) - len(raw.lstrip())
            if indent <= block_indent:
                in_block = False
                continue
            if stripped.startswith(':'):
                continue
            ident = _IDENTIFIER_RE.match(stripped)
            if ident:
                names.add(ident.group(1).rsplit('.', 1)[-1])
    return names


@pytest.fixture(scope='module')
def public_symbols() -> set[str]:
    return _collect_public_symbols()


@pytest.fixture(scope='module')
def documented_names() -> set[str]:
    return _collect_documented_names()


def test_public_api_is_documented(public_symbols, documented_names):
    """Every public ``pyvista`` class/function must appear in the Sphinx reference."""
    missing = public_symbols - documented_names - _ALLOWED_UNDOCUMENTED
    assert not missing, (
        f'{len(missing)} public pyvista symbol(s) are not referenced under '
        f'doc/source/api/. Add each to an autosummary block, or - if the '
        f'symbol is an internal helper that should stay out of the reference '
        f'- add it to _ALLOWED_UNDOCUMENTED with a one-line rationale:\n  '
        + '\n  '.join(sorted(missing))
    )


def test_allowlist_stays_accurate(public_symbols, documented_names):
    """Flag stale entries so ``_ALLOWED_UNDOCUMENTED`` does not rot silently."""
    stale_removed = _ALLOWED_UNDOCUMENTED - public_symbols
    stale_documented = _ALLOWED_UNDOCUMENTED & documented_names
    problems = []
    if stale_removed:
        problems.append(
            'No longer public (drop from allowlist): ' + ', '.join(sorted(stale_removed))
        )
    if stale_documented:
        problems.append(
            'Now documented (drop from allowlist): ' + ', '.join(sorted(stale_documented))
        )
    assert not problems, '\n'.join(problems)
