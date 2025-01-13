from __future__ import annotations

import subprocess
import sys

import pytest


def _module_is_loaded(module: str) -> bool:
    """This function checks if the specified module is loaded after calling `import pyvista`

    We use ``os.system`` because we need to test the import of pyvista
    outside of the pytest unit test framework as pytest loads vtk.
    """
    exe_str = f"import pyvista; import sys; assert '{module}' not in sys.modules"

    # Run the Python command in a subprocess
    result = subprocess.run([sys.executable, '-c', exe_str], capture_output=True, text=True)  # noqa: S603

    # Return True if the assertion failed (non-zero return code)
    return result.returncode != 0


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
