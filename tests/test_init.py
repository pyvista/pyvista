from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _module_is_loaded(module: str) -> bool:
    """This function checks if the specified module is loaded after calling `import pyvista`

    We use ``os.system`` because we need to test the import of pyvista
    outside of the pytest unit test framework as pytest loads vtk.
    """
    exe_str = f"import pyvista; import sys; assert '{module}' not in sys.modules"

    # anything other than 0 indicates the assertion raised
    return os.system(f'{sys.executable} -c "{exe_str}"') != 0


def test_vtk_not_loaded():
    error_msg = """
    vtk has been directly imported in vtk>=9
    Please see:
    https://github.com/pyvista/pyvista/pull/1163
    """
    assert not _module_is_loaded('vtk'), error_msg


@pytest.mark.parametrize('module', ['matplotlib', 'scipy'])
def test_large_dependencies_not_imported(module: str):
    error_msg = f"""
    Module `{module}` was loaded at root `import pyvista`.
    This can drastically slow down initial import times.
    Please see
    https://github.com/pyvista/pyvista/pull/7023
    """
    assert not _module_is_loaded(module), error_msg


def test_pyvista_oo_flag():
    """Test that PyVista works correctly with the -OO optimization flag."""
    code = 'from pyvista import Chart2D'

    command = [sys.executable, '-OO', '-c', code]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f'PyVista failed with -OO flag. stderr: {result.stderr}'
