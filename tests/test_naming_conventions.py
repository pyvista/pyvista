from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import pyvista as pv

# Project root (pyvista package directory)
project_root = Path(pv.__file__).parent.parent

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


@pytest.mark.skip_windows('Needs grep')
def test_no_bare_pyvista_imports():
    # Allow leading whitespace:
    #   import pyvista
    pattern = r'^[[:space:]]*import[[:space:]]+pyvista[[:space:]]*$'

    result = subprocess.run(
        ['grep', *COMMON_GREP_ARGS, pattern, project_root],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "Found bare 'import pyvista' imports, "
        "use 'import pyvista as pv' instead\n\n"
        f'{result.stdout}'
    )


FORBIDDEN_NAMES = ['plotter', 'p', 'plot']


@pytest.mark.skip_windows('Needs grep')
def test_no_forbidden_plotter_names():
    names = '|'.join(FORBIDDEN_NAMES)
    pattern = (
        rf'^[[:space:]]*({names})[[:space:]]*='
        rf'[[:space:]]*pv\.Plotter\('
    )

    result = subprocess.run(
        ['grep', *COMMON_GREP_ARGS, pattern, project_root],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        f"Found forbidden Plotter variable names (use 'pl' instead):\n\n{result.stdout}"
    )
