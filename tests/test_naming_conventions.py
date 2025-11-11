from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import pyvista as pv

PROJECT_ROOT = Path(pv.__file__).parent.parent
TESTS_ROOT = PROJECT_ROOT / 'tests'
assert TESTS_ROOT.is_dir()

# Common grep arguments
COMMON_GREP_ARGS = [
    '-R',
    '-E',  # use extended regex everywhere
    '--include=*.py',
    '--include=*.rst',
    '--include=*.md',
    '--binary-files=without-match',
    '--exclude-dir=*site-packages*',
    '--exclude-dir=__pycache__',
    '--exclude-dir=vtk-data',
]


def _run_grep(args, pattern, path):
    return subprocess.run(
        ['grep', *args, pattern, path],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skip_windows('Needs grep')
def test_no_bare_vtk_imports_in_tests():
    # Search for `import vtk` or `from vtk`
    pattern = r'^[[:space:]]*(import[[:space:]]+vtk\b|from[[:space:]]+vtk\b)'
    result = _run_grep(COMMON_GREP_ARGS, pattern, TESTS_ROOT)
    assert result.returncode != 0, (
        "Found bare 'import vtk' or `from vtk` imports, import vtk from pyvista instead, e.g. "
        '`from pyvista.core import _vtk_core as _vtk`\n\n'
        f'{result.stdout}'
    )


@pytest.mark.skip_windows('Needs grep')
def test_no_bare_pyvista_imports_in_project():
    # Search for `import pyvista`
    pattern = r'^[[:space:]]*import[[:space:]]+pyvista[[:space:]]*$'
    result = _run_grep(COMMON_GREP_ARGS, pattern, PROJECT_ROOT)
    assert result.returncode != 0, (
        "Found bare 'import pyvista' imports, "
        "use 'import pyvista as pv' instead\n\n"
        f'{result.stdout}'
    )


@pytest.mark.skip_windows('Needs grep')
def test_no_forbidden_plotter_names_in_project():
    # Search `name = pv.Plotter(` with forbidden name
    forbidden_names = ['plotter', 'p', 'plot']
    names = '|'.join(forbidden_names)
    pattern = (
        rf'^[[:space:]]*({names})[[:space:]]*='
        rf'[[:space:]]*pv\.Plotter\('
    )
    result = _run_grep(COMMON_GREP_ARGS, pattern, PROJECT_ROOT)
    assert result.returncode != 0, (
        f"Found forbidden Plotter variable names (use 'pl' instead):\n\n{result.stdout}"
    )
