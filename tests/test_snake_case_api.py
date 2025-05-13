from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import pyvista as pv
from pyvista.core._vtk_core import DisableVtkSnakeCase
from pyvista.core.errors import PyVistaAttributeError
from pyvista.core.errors import VTKVersionError


def get_all_pyvista_classes() -> tuple[tuple[str, ...], tuple[type, ...]]:
    """Return all classes defined in the pyvista package."""
    class_types: set[type] = set()

    package_path = Path(pv.__path__[0])  # path to pyvista package

    def find_py_files(path: Path):
        return [p for p in path.rglob('*.py') if p.name != '__init__.py']

    core_py_files = find_py_files(package_path / 'core')
    plotting_py_files = find_py_files(package_path / 'plotting')

    for file_path in [*core_py_files, *plotting_py_files]:
        # Convert path to dotted module name
        rel_path = file_path.relative_to(package_path.parent)
        module_name = '.'.join(rel_path.with_suffix('').parts)

        module = importlib.import_module(module_name)

        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and getattr(obj, '__module__', '') == module_name:
                class_types.add(obj)

    # Sort and return names and types as separate tuples
    sorted_classes = sorted(class_types, key=lambda cls: cls.__name__)
    return tuple(cls.__name__ for cls in sorted_classes), tuple(sorted_classes)


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'vtk_subclass' in metafunc.fixturenames:
        class_names, class_types = get_all_pyvista_classes()

        def inherits_from_vtk(klass):
            bases = klass.__mro__[1:]
            return any(base.__name__.startswith('vtk') for base in bases)

        filtered = {
            name: cls for name, cls in zip(class_names, class_types) if inherits_from_vtk(cls)
        }
        assert filtered
        names = filtered.keys()
        types = filtered.values()
        metafunc.parametrize('vtk_subclass', types, ids=names)


def try_init_object(class_, kwargs):
    # Init object but skip if abstract
    try:
        instance = class_(**kwargs)
    except VTKVersionError:
        pytest.skip('VTK Version not supported.')
    except TypeError as e:
        if 'abstract' in repr(e):
            pytest.skip('Class is abstract.')
        raise
    return instance


def test_vtk_snake_case_api_is_disabled(vtk_subclass):
    # Define kwargs as required for some cases.
    assert pv.vtk_snake_case() == 'error'
    kwargs = {}
    if vtk_subclass is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif vtk_subclass is pv.Renderer:
        kwargs['parent'] = pv.Plotter()
    elif vtk_subclass in [pv.ChartBox, pv.ChartPie]:
        kwargs['data'] = list(range(10))
    elif vtk_subclass is pv.CompositeAttributes:
        kwargs['mapper'] = pv.CompositePolyDataMapper()
        kwargs['dataset'] = pv.MultiBlock()
    elif vtk_subclass is pv.CornerAnnotation:
        kwargs['position'] = 'top'
        kwargs['text'] = 'text'
    elif vtk_subclass is pv.plotting.utilities.algorithms.ActiveScalarsAlgorithm:
        kwargs['name'] = 'name'
    elif vtk_subclass is pv.charts.AreaPlot:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['y1'] = (1, 0, 0)
    elif vtk_subclass in [pv.charts.BarPlot, pv.charts.LinePlot2D, pv.charts.ScatterPlot2D]:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['y'] = (1, 0, 0)
    elif vtk_subclass is pv.charts.StackPlot:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['ys'] = (1, 0, 0)
    elif vtk_subclass in [pv.charts.BoxPlot, pv.charts.PiePlot]:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['data'] = [0, 0, 0]
    elif vtk_subclass is pv.charts._ChartBackground:
        kwargs['chart'] = pv.charts.Chart2D()
    elif vtk_subclass is pv.plotting.background_renderer.BackgroundRenderer:
        kwargs['parent'] = pv.Plotter()
        kwargs['image_path'] = pv.examples.logofile

    instance = try_init_object(vtk_subclass, kwargs)
    vtk_attr_camel_case = 'GetGlobalWarningDisplay'
    vtk_attr_snake_case = 'global_warning_display'

    # Make sure the CamelCase attribute exists and can be accessed
    assert hasattr(instance, vtk_attr_camel_case)

    if pv.vtk_version_info >= (9, 4):
        # Test getting the snake_case equivalent raises error

        try:
            getattr(instance, vtk_attr_snake_case)
        except PyVistaAttributeError as e:
            # Test passes, we want an error to be raised
            # Confirm error message is correct
            match = "The attribute 'global_warning_display' is defined by VTK and is not part of the PyVista API"
            assert match in repr(e)  # noqa: PT017
        else:
            if DisableVtkSnakeCase not in vtk_subclass.__mro__:
                msg = (
                    f'The class {vtk_subclass.__name__!r} in {vtk_subclass.__module__!r}\n'
                    f'must inherit from {DisableVtkSnakeCase.__name__!r} in {DisableVtkSnakeCase.__module__!r}'
                )
            else:
                msg = f'{PyVistaAttributeError.__name__} was NOT raised (but was expected).'
            pytest.fail(msg)
    else:
        assert not hasattr(instance, vtk_attr_snake_case)
