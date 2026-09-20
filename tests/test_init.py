from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import pytest

from pyvista import _vtk
from tests.vtk_backend_divergence import CVISTA_NAMESPACE

CORE_VTKMODULES = {
    'vtkmodules.numpy_interface',
    'vtkmodules.numpy_interface._vtk_array_mixin',
    'vtkmodules.numpy_interface.array_overrides',
    'vtkmodules.numpy_interface.dataset_adapter',
    'vtkmodules.numpy_interface.utils',
    'vtkmodules.numpy_interface.vtk_affine_array',
    'vtkmodules.numpy_interface.vtk_aos_array',
    'vtkmodules.numpy_interface.vtk_composite_array',
    'vtkmodules.numpy_interface.vtk_constant_array',
    'vtkmodules.numpy_interface.vtk_implicit_array',
    'vtkmodules.numpy_interface.vtk_indexed_array',
    'vtkmodules.numpy_interface.vtk_none_array',
    'vtkmodules.numpy_interface.vtk_partitioned_array',
    'vtkmodules.numpy_interface.vtk_soa_array',
    'vtkmodules.numpy_interface.vtk_strided_array',
    'vtkmodules.numpy_interface.vtk_structured_point_array',
    'vtkmodules.util',
    'vtkmodules.util.data_array_selection',
    'vtkmodules.util.data_model',
    'vtkmodules.util.execution_model',
    'vtkmodules.util.graph',
    'vtkmodules.util.implicit_functions',
    'vtkmodules.util.information',
    'vtkmodules.util.matrix',
    'vtkmodules.util.molecule',
    'vtkmodules.util.numpy_support',
    'vtkmodules.util.pickle_support',
    'vtkmodules.util.selection',
    'vtkmodules.util.string_array',
    'vtkmodules.util.variant_array',
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
    'vtkmodules.vtkRenderingContextOpenGL2',
    'vtkmodules.vtkRenderingCore',
    'vtkmodules.vtkRenderingFreeType',
    'vtkmodules.vtkRenderingHyperTreeGrid',
    'vtkmodules.vtkRenderingLabel',
    'vtkmodules.vtkRenderingMatplotlib',
    'vtkmodules.vtkRenderingOpenGL2',
    'vtkmodules.vtkRenderingUI',
    'vtkmodules.vtkRenderingVolume',
    'vtkmodules.vtkRenderingVolumeOpenGL2',
}


def exec_success(code: str):
    return subprocess.run([sys.executable, '-c', code], check=False).returncode == 0


def _module_is_loaded(module_to_check: str, module_to_import: str = 'pyvista') -> bool:
    """This function checks if the specified module is loaded after calling `import pyvista`

    We use a subprocess because we need to test the import of pyvista
    outside of the pytest unit test framework as pytest loads vtk.
    """
    exe_str = (
        f"import {module_to_import}; import sys; assert '{module_to_check}' not in sys.modules"
    )

    # anything other than 0 indicates the assertion raised
    return not exec_success(exe_str)


@pytest.mark.parametrize(
    ('allowed_modules', 'module_to_import'),
    [(CORE_VTKMODULES, 'pyvista'), (PLOTTING_VTKMODULES, 'pyvista.plotting')],
    ids=['core', 'plotting'],
)
def test_minimal_vtkmodules_imported(allowed_modules, module_to_import):
    # Import in a fresh interpreter, since pytest itself has already loaded VTK here
    code = (
        f'import {module_to_import}, sys; '
        "print(*(m for m in sys.modules if m.startswith('vtkmodules.')))"
    )
    imported = subprocess.run(
        [sys.executable, '-c', code], check=True, capture_output=True, text=True
    ).stdout.split()
    vtkmodules_loaded = set(imported) - allowed_modules

    error_msg = """
    Disallowed VTK module(s) were loaded at root `import pyvista`.
    This can drastically slow down initial import times.
    """
    assert sorted(vtkmodules_loaded) == [], error_msg


@pytest.mark.parametrize('module', ['PIL', 'matplotlib', 'scipy'])
def test_large_dependencies_not_imported(module: str):
    error_msg = f"""
    Module `{module}` was loaded at root `import pyvista`.
    This can drastically slow down initial import times.
    Please see
    https://github.com/pyvista/pyvista/pull/7023
    """
    assert not _module_is_loaded(module), error_msg


def test_plotting_import_has_no_direct_pillow_imports():
    """Pillow must stay behind a local import in every plotting module.

    This asks who *performed* the import rather than whether ``PIL`` is in
    ``sys.modules``, because ``import pyvista.plotting`` pulls in matplotlib,
    which imports Pillow itself. A ``sys.modules`` check would therefore be red
    no matter what PyVista does, and the root-import check in
    ``test_large_dependencies_not_imported`` cannot see a plotting module at
    all.

    Both ``import PIL`` and ``importlib.import_module('PIL')`` are watched; the
    latter does not go through ``builtins.__import__``, so hooking only that
    leaves a way to write the defect and still pass.
    """
    code = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        original_import = builtins.__import__
        original_import_module = importlib.import_module
        pyvista_pillow_imports = []

        def record(caller_name, name):
            if caller_name.startswith('pyvista') and (
                name == 'PIL' or name.startswith('PIL.')
            ):
                pyvista_pillow_imports.append((caller_name, name))

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            record(globals.get('__name__', '') if globals else '', name)
            return original_import(name, globals, locals, fromlist, level)

        def tracked_import_module(name, package=None):
            record(sys._getframe(1).f_globals.get('__name__', ''), name)
            return original_import_module(name, package)

        builtins.__import__ = tracked_import
        importlib.import_module = tracked_import_module
        import pyvista.plotting  # noqa: F401

        assert pyvista_pillow_imports == [], pyvista_pillow_imports
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_pyvista_oo_flag():
    """Test that PyVista works correctly with the -OO optimization flag."""
    code = 'from pyvista import Chart2D'

    command = [sys.executable, '-OO', '-c', code]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f'PyVista failed with -OO flag. stderr: {result.stderr}'


@pytest.mark.skip_vtk_backend('cvista', reason=CVISTA_NAMESPACE)
def test_plotting_import_loads_context_opengl2():
    code = (
        'import pyvista.plotting\n'
        'import sys\n'
        "assert 'vtkmodules.vtkRenderingContextOpenGL2' in sys.modules"
    )
    assert exec_success(code)


def test_plotting_attribute_cached():
    # Lazy-loaded plotting attributes should be cached after first access
    exe_str = (
        'import pyvista as pv\n'
        'import sys\n'
        "assert 'Plotter' not in pv.__dict__\n"
        '_ = pv.Plotter\n'
        "assert 'Plotter' in pv.__dict__"
    )
    assert exec_success(exe_str)


def test_plotting_getattr_called_once():
    # pyvista.__getattr__ should only be called once for a lazy-loaded plotting attribute
    exe_str = (
        'import pyvista as pv\n'
        "counter = {'count': 0}\n"
        '\n'
        'original_getattr = pv.__getattr__\n'
        'def wrapped(name):\n'
        "    if name == 'Plotter':\n"
        "       counter['count'] += 1\n"
        '    return original_getattr(name)\n'
        'pv.__getattr__ = wrapped\n'
        '\n'
        "assert counter['count'] == 0\n"
        '_ = pv.Plotter\n'
        "assert counter['count'] == 1\n"
        '_ = pv.Plotter\n'
        "assert counter['count'] == 1\n"
    )
    assert exec_success(exe_str)


def _loader():
    """Dummy special loader."""
    return object()


def test_vtk_import_all_suppressed(monkeypatch):
    monkeypatch.setattr(_vtk, '_VTK_CLASS_TO_MODULE', {'A': 'vtkFoo', 'B': 'vtkBar'})
    monkeypatch.setattr(_vtk, '_SPECIAL_LOADERS', {'SpecialA': _loader, 'SpecialB': _loader})

    calls = []

    def fake_has_attr(name):
        calls.append(name)
        return True

    monkeypatch.setattr(_vtk, 'has_attr', fake_has_attr)

    _vtk.import_all(suppress_import_errors=True)

    assert calls == ['A', 'B', 'SpecialA', 'SpecialB']


def test_vtk_import_all_not_suppressed(monkeypatch):
    monkeypatch.setattr(_vtk, '_VTK_CLASS_TO_MODULE', {'A': 'vtkFoo', 'B': 'vtkBar'})
    monkeypatch.setattr(_vtk, '_SPECIAL_LOADERS', {'SpecialA': _loader, 'SpecialB': _loader})

    calls = []

    def fake_getattr(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(_vtk, '__getattr__', fake_getattr)

    _vtk.import_all(suppress_import_errors=False)

    assert calls == ['A', 'B', 'SpecialA', 'SpecialB']


def test_vtk_import_all_not_suppressed_propagates(monkeypatch):
    monkeypatch.setattr(_vtk, '_VTK_CLASS_TO_MODULE', {'A': 'vtkFoo'})
    monkeypatch.setattr(_vtk, '_SPECIAL_LOADERS', {'SpecialA': _loader})

    def fake_getattr(name):  # noqa: ARG001
        msg = 'boom'
        raise ImportError(msg)

    monkeypatch.setattr(_vtk, '__getattr__', fake_getattr)

    with pytest.raises(ImportError, match='boom'):
        _vtk.import_all(suppress_import_errors=False)


def test_vtk_import_all_suppressed_ignores_failures(monkeypatch):
    monkeypatch.setattr(_vtk, '_VTK_CLASS_TO_MODULE', {'A': 'vtkFoo'})
    monkeypatch.setattr(_vtk, '_SPECIAL_LOADERS', {'SpecialA': _loader})

    calls = []

    def fake_has_attr(name):
        calls.append(name)
        return False

    monkeypatch.setattr(_vtk, 'has_attr', fake_has_attr)

    _vtk.import_all(suppress_import_errors=True)

    assert calls == ['A', 'SpecialA']


@pytest.mark.parametrize(('building_gallery', 'imported'), [('true', True), ('false', False)])
def test_building_gallery_imports_vtk_eagerly(building_gallery, imported):
    """A gallery build resolves every mapped VTK class when PyVista is imported."""
    code = "from pyvista import _vtk; print('vtkOutlineFilter' in vars(_vtk))"
    result = subprocess.run(
        [sys.executable, '-c', code],
        env={**os.environ, 'PYVISTA_BUILDING_GALLERY': building_gallery},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(imported)


def test_validation_forward_deprecated():
    import pyvista_validation

    import pyvista as pv

    msg = (
        '`pyvista._validation` has moved to the `pyvista_validation` package; '
        'use `from pyvista_validation import ...` instead.'
    )
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(msg)):
        assert pv._validation is pyvista_validation
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(msg)):
        from pyvista import _validation
    assert _validation.validate_array3([1, 2, 3]).shape == (3,)
