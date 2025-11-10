from __future__ import annotations

import subprocess

import pytest


@pytest.mark.skip_windows('Needs grep')
def test_no_bare_pyvista_imports():
    result = subprocess.run(
        [
            'grep',
            '-R',
            '--include=*.py',
            '--binary-files=without-match',
            r'^import pyvista\s*$',
            '.',
            '--exclude-dir=*site-packages*',
            '--exclude-dir=__pycache__',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Found bare 'import pyvista' imports, "
        f"use 'import pyvista as pv' instead\n\n{result.stdout}"
    )
