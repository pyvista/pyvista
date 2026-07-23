"""Tests for the ``download_examples`` Sphinx extension.

This extension is deliberately independent of ``plot_directive.py`` (it
doesn't import anything from it, and works whether or not that extension is
even installed), so its tests get their own small, self-contained fixture
directory (``tinypages_download_examples/``) with its own ``conf.py``,
rather than living inside the main ``tinypages/`` used by
``test_tinypages.py``. That fixture's ``pyvista_plot_directive/`` output is
checked against exact, hash-locked file sets for both serial and parallel
builds; a separate fixture avoids needing to update those for a feature
that has nothing to do with them.

The one thing worth checking for *integration* with the real docs build --
that ``pyvista.ext.download_examples`` is wired into the real ``tinypages/``
``conf.py`` and produces a download for at least one real docstring -- lives
in ``test_tinypages.py`` instead, alongside the rest of that build's checks.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyvista.plotting import system_supports_plotting
from tests.conftest import flaky_test
from tests.plotting.test_tinypages import _run_sphinx_build
from tests.plotting.test_tinypages import _sphinx_build_cmd

pytest.importorskip('sphinx')

if not system_supports_plotting():
    pytestmark = pytest.mark.skip(reason='Requires system to support plotting')

SRCDIR = Path(__file__).parent / 'tinypages_download_examples'


def _read(paths: list[Path], name_contains: str) -> str:
    """Find the one generated file whose name contains ``name_contains`` and return its text."""
    matches = [p for p in paths if name_contains in p.name]
    assert matches, f'No generated file matching {name_contains!r} in {[p.name for p in paths]}'
    assert len(matches) == 1, f'Expected exactly one match for {name_contains!r}, got {matches}'
    return matches[0].read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def built(tmp_path_factory) -> tuple[Path, list[Path]]:
    """Build the fixture once and share it (and its generated files) across this module."""
    tmp_path = tmp_path_factory.mktemp('download_examples_build')
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'

    returncode, out, err = _run_sphinx_build(
        _sphinx_build_cmd(SRCDIR, html_dir, doctree_dir),
    )
    assert returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    downloads_dir = html_dir / '_downloads'
    examples = sorted(downloads_dir.rglob('*.py')) if downloads_dir.is_dir() else []
    return html_dir, examples


@flaky_test(exceptions=(AssertionError,))
def test_download_examples_execute(built: tuple[Path, list[Path]]):
    """Every generated example script should run standalone without error.

    This is the key correctness check: a script that merely *looks*
    plausible would still fail here if, say, it were missing an import
    that got silently dropped during conversion.
    """
    _html_dir, examples = built
    assert examples, 'expected at least one generated example script'

    env = {**os.environ, 'PYVISTA_OFF_SCREEN': 'true'}
    failures = []
    for path in examples:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=path.parent,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f'{path.name}:\n{result.stdout}\n{result.stderr}')

    assert not failures, 'Some generated examples failed to execute:\n\n' + '\n\n'.join(failures)


def test_no_examples_section_produces_no_download(built: tuple[Path, list[Path]]):
    """A docstring with no "Examples" heading at all should be untouched."""
    _html_dir, examples = built
    assert not any('case_no_examples' in p.name for p in examples)


def test_prose_only_examples_produces_no_download(built: tuple[Path, list[Path]]):
    """An Examples section with no real code should not produce a download."""
    _html_dir, examples = built
    assert not any('case_prose_only' in p.name for p in examples)


def test_pyvista_plot_wrapped_examples_still_convert(built: tuple[Path, list[Path]]):
    """A pyvista-importing Examples section gets auto-wrapped in ``.. pyvista-plot::``.

    That directive wraps its generated source in a
    ``.. container:: pyvista-plot-source`` node (see plot_directive.py's
    TEMPLATE) -- this confirms the Examples-heading scan isn't thrown off
    by that extra container, still extracts the real code from inside it,
    and drops the rendered figure rather than turning it into a comment.
    """
    _html_dir, examples = built
    src = _read(examples, 'case_pyvista_plot_wrapped')
    assert 'import pyvista as pv' in src
    assert 'pv.Sphere().plot()' in src
    # nothing figure/image-related should have leaked in as a comment
    assert 'image' not in src.lower()
    assert '.png' not in src


def test_docstring_examples_conversion(built: tuple[Path, list[Path]]):
    """Spot-check the prose/code conversion rules on a few representative cases."""
    _html_dir, examples = built

    dropdown_src = _read(examples, 'case_dropdown')
    assert 'Click me' not in dropdown_src
    assert 'hidden content' not in dropdown_src.lower()
    assert 'import sys' in dropdown_src

    tabset_src = _read(examples, 'case_tabset')
    assert 'Static Scene' not in tabset_src
    assert 'Interactive Scene' not in tabset_src
    assert 'iframe' not in tabset_src

    note_src = _read(examples, 'case_note')
    assert '# NOTE:' in note_src
    assert '# This is a note' in note_src

    warning_src = _read(examples, 'case_warning')
    assert '# WARNING:' in warning_src

    multi_note_src = _read(examples, 'case_multi_paragraph_note')
    assert '# First paragraph of the note.' in multi_note_src
    assert '# Second paragraph of the note.' in multi_note_src

    admonition_src = _read(examples, 'case_generic_admonition')
    assert '# Custom Title:' in admonition_src

    xref_src = _read(examples, 'case_xref_plain')
    assert '`docstring_cases.Sample`' in xref_src
    assert '`docstring_cases.Sample.show()`' in xref_src

    xref_title_src = _read(examples, 'case_xref_explicit_title')
    assert '`Sample class`' in xref_title_src

    ref_src = _read(examples, 'case_ref_plain')
    # a plain :ref: keeps its resolved display text with no backticks
    assert '`' not in ref_src.split('import sys')[0]

    inline_literal_src = _read(examples, 'case_inline_literal')
    assert '`some_variable = True`' in inline_literal_src

    combined_src = _read(examples, 'case_combined')
    assert '`docstring_cases.Sample`' in combined_src
    assert '`some_variable = True`' in combined_src
    assert '# NOTE:' in combined_src
    assert 'More details' not in combined_src  # dropped dropdown
    assert '# 3' in combined_src  # doctest output line becomes a comment
