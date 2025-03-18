from __future__ import annotations

import os
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


def test_failed():
    pytest.fail('test')


@pytest.mark.skipif(os.name == 'nt', reason='Test skipped on Windows')
def test_failed_windows():
    pytest.fail('test')


@pytest.mark.skipif(sys.version_info >= (3, 11), reason='Test skipped for python version >= 3.11')
def test_failed_python_version():
    pytest.fail('test')


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
