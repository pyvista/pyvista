from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista.core.errors import VTKVersionError

if TYPE_CHECKING:
    from types import ModuleType


def _get_classes_with_bounds(module: ModuleType) -> list[type]:
    return [
        getattr(module, item)
        for item in module.__dict__.keys()
        if hasattr(getattr(module, item), 'bounds')
    ]


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'class_with_bounds' in metafunc.fixturenames:
        # Generate a separate test for any class that has bounds
        core_classes: list[type] = _get_classes_with_bounds(pv.core)
        plotting_classes: list[type] = _get_classes_with_bounds(pv.plotting)
        classes = [*core_classes, *plotting_classes]
        class_names = [class_.__name__ for class_ in classes]

        # List all the classes explicitly
        expected_names = [
            'Actor',
            'AxesAssembly',
            'AxesAssemblySymmetric',
            'BasePlotter',
            'BoxSource',
            'Cell',
            'CompositePolyDataMapper',
            'CubeAxesActor',
            'CubeSource',
            'DataSet',
            'DataSetMapper',
            'ExplicitStructuredGrid',
            'FixedPointVolumeRayCastMapper',
            'GPUVolumeRayCastMapper',
            'Grid',
            'ImageData',
            'Label',
            'MultiBlock',
            'OpenGLGPUVolumeRayCastMapper',
            'OrthogonalPlanesSource',
            'PlanesAssembly',
            'Plotter',
            'PointGaussianMapper',
            'PointGrid',
            'PointSet',
            'PolyData',
            'Prop3D',
            'RectilinearGrid',
            'Renderer',
            'SmartVolumeMapper',
            'StructuredGrid',
            'UnstructuredGrid',
            'UnstructuredGridVolumeRayCastMapper',
            'Volume',
        ]
        assert sorted(class_names) == sorted(expected_names)

        metafunc.parametrize('class_with_bounds', classes, ids=class_names)


def get_property_return_type(prop: property):
    members = inspect.getmembers(prop)
    for member in members:
        name, func = member
        if name == 'fget':
            return func.__annotations__['return']
    return None


def test_bounds_tuple(class_with_bounds):
    # Define kwargs as required for some cases.
    kwargs = {}
    if class_with_bounds is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif class_with_bounds is pv.Renderer:
        kwargs['parent'] = pv.Plotter()

    # Init object but skip if abstract
    try:
        instance = class_with_bounds(**kwargs)
    except VTKVersionError:
        pytest.skip('VTK Version not supported.')
    except TypeError as e:
        if 'abstract' in repr(e):
            pytest.skip('Class is abstract.')
        raise

    # Do test
    bounds = instance.bounds
    assert len(bounds) == 6
    assert isinstance(bounds, pv.BoundsTuple)
    # Make sure we have built-in floats
    assert all(isinstance(bnd, float) and not isinstance(bnd, np.generic) for bnd in bounds)

    # Test type annotations
    return_type = get_property_return_type(class_with_bounds.bounds)
    assert return_type == 'BoundsTuple'
