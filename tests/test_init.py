from __future__ import annotations

import os
import subprocess
import sys

import pytest

CORE_VTKMODULES = {
    'vtkmodules.numpy_interface',
    'vtkmodules.numpy_interface.dataset_adapter',
    'vtkmodules.util',
    'vtkmodules.util.data_model',
    'vtkmodules.util.execution_model',
    'vtkmodules.util.numpy_support',
    'vtkmodules.util.pickle_support',
    'vtkmodules.util.vtkConstants',
    'vtkmodules.vtkCommonCore',
    'vtkmodules.vtkCommonDataModel',
    'vtkmodules.vtkCommonExecutionModel',
    'vtkmodules.vtkCommonMath',
    'vtkmodules.vtkCommonMisc',
    'vtkmodules.vtkCommonSystem',
    'vtkmodules.vtkCommonTransforms',
    'vtkmodules.vtkFiltersCore',
    'vtkmodules.vtkFiltersSources',
    'vtkmodules.vtkImagingSources',
    'vtkmodules.vtkParallelCore',
}
PLOTTING_VTKMODULES = CORE_VTKMODULES | {
    'vtkmodules.util.vtkAlgorithm',
    'vtkmodules.vtkChartsCore',
    'vtkmodules.vtkCommonColor',
    'vtkmodules.vtkFiltersGeneral',
    'vtkmodules.vtkFiltersPython',
    'vtkmodules.vtkIOCore',
    'vtkmodules.vtkIOImage',
    'vtkmodules.vtkImagingCore',
    'vtkmodules.vtkImagingMath',
    'vtkmodules.vtkInteractionStyle',
    'vtkmodules.vtkPythonContext2D',
    'vtkmodules.vtkRenderingAnnotation',
    'vtkmodules.vtkRenderingContext2D',
    'vtkmodules.vtkRenderingCore',
    'vtkmodules.vtkRenderingFreeType',
    'vtkmodules.vtkRenderingHyperTreeGrid',
    'vtkmodules.vtkRenderingMatplotlib',
    'vtkmodules.vtkRenderingOpenGL2',
    'vtkmodules.vtkRenderingUI',
    'vtkmodules.vtkRenderingVolume',
    'vtkmodules.vtkRenderingVolumeOpenGL2',
}


def _module_is_loaded(module_to_check: str, module_to_import: str = 'pyvista') -> bool:
    """This function checks if the specified module is loaded after calling `import pyvista`

    We use ``os.system`` because we need to test the import of pyvista
    outside of the pytest unit test framework as pytest loads vtk.
    """
    exe_str = (
        f"import {module_to_import}; import sys; assert '{module_to_check}' not in sys.modules"
    )

    # anything other than 0 indicates the assertion raised
    return os.system(f'{sys.executable} -c "{exe_str}"') != 0


@pytest.mark.parametrize(
    ('allowed_modules', 'module_to_import'),
    [(CORE_VTKMODULES, 'pyvista'), (PLOTTING_VTKMODULES, 'pyvista.plotting')],
    ids=['core', 'plotting'],
)
def test_minimal_vtkmodules_imported(allowed_modules, module_to_import):
    vtkmodules_not_allowed = sorted(
        {
            module
            for module in sys.modules
            if module.startswith('vtkmodules.') and module not in allowed_modules
        }
    )
    vtkmodules_loaded = {
        module
        for module in vtkmodules_not_allowed
        if _module_is_loaded(module_to_import=module_to_import, module_to_check=module)
    }

    error_msg = """
    Disallowed VTK module(s) were loaded at root `import pyvista`.
    This can drastically slow down initial import times.
    """
    assert sorted(vtkmodules_loaded) == [], error_msg


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
