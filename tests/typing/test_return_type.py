from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import VTKVersionError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType


def get_classes_with_attribute(attr: str) -> tuple[tuple[str], tuple[type]]:
    """Return all classes (type and name) with a specific attribute."""
    class_types: list[type] = []

    def _get_classes_from_module(module: ModuleType) -> list[type]:
        keys = module.__dict__.keys()
        for module_item in keys:
            module_attr = getattr(module, module_item)
            try:
                issubclass(module_attr, object)
            except TypeError:
                pass  # not a class
            else:
                if hasattr(module_attr, attr):
                    class_types.append(module_attr)

    # Get from core and plotting separately since plotting module has a lazy importer
    _get_classes_from_module(pv.core)
    _get_classes_from_module(pv.plotting)

    # Sort and return names and types as separate tuples
    dict_ = {class_.__name__: class_ for class_ in class_types}
    sorted_dict = {key: dict_[key] for key in sorted(dict_.keys())}
    return tuple(sorted_dict.keys()), tuple(sorted_dict.values())


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'class_with_bounds' in metafunc.fixturenames:
        # Generate a separate test for any class that has bounds
        class_names, class_types = get_classes_with_attribute('bounds')
        metafunc.parametrize('class_with_bounds', class_types, ids=class_names)

    if 'class_with_center' in metafunc.fixturenames:
        # Generate a separate test for any class that has a center
        class_names, class_types = get_classes_with_attribute('center')
        metafunc.parametrize('class_with_center', class_types, ids=class_names)


def is_all_floats(iterable: Iterable):
    """Return True if input only has built-in floats."""
    return all(isinstance(item, float) and not isinstance(item, np.generic) for item in iterable)


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


def get_property_return_type(prop: property):
    members = inspect.getmembers(prop)
    for member in members:
        name, func = member
        if name == 'fget':
            return func.__annotations__['return']
    return None


def test_bounds_tuple(class_with_bounds, catch_vtk_errors):
    if _vtk.is_vtk_attribute(class_with_bounds, 'bounds'):
        pytest.skip('bounds is defined by vtk, not pyvista.')
    if class_with_bounds is pv.DataSetMapper:
        catch_vtk_errors.skip = True

    # Define kwargs as required for some cases.
    kwargs = {}
    if class_with_bounds is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif class_with_bounds is pv.Renderer:
        kwargs['parent'] = pv.Plotter()

    instance = try_init_object(class_with_bounds, kwargs)

    # Test type at runtime
    bounds = instance.bounds
    assert len(bounds) == 6
    assert isinstance(bounds, pv.BoundsTuple)
    assert is_all_floats(bounds)

    # Test type annotations
    return_type = get_property_return_type(class_with_bounds.bounds)
    assert return_type == 'BoundsTuple'


def test_bounds_size(class_with_bounds, catch_vtk_errors):
    if _vtk.is_vtk_attribute(class_with_bounds, 'bounds'):
        pytest.skip('bounds is defined by vtk, not pyvista.')
    elif class_with_bounds.__name__.endswith('Source'):
        pytest.skip('Source objects use bounds as setters.')
    if class_with_bounds is pv.DataSetMapper:
        catch_vtk_errors.skip = True

    # Define kwargs as required for some cases.
    kwargs = {}
    if class_with_bounds is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif class_with_bounds is pv.Renderer:
        kwargs['parent'] = pv.Plotter()

    instance = try_init_object(class_with_bounds, kwargs)

    # Test type at runtime
    bounds_size = instance.bounds_size
    assert len(bounds_size) == 3
    assert isinstance(bounds_size, tuple)
    assert is_all_floats(bounds_size)

    # Test type annotations
    return_type = get_property_return_type(class_with_bounds.bounds_size)
    assert return_type == 'tuple[float, float, float]'


def test_center_tuple(class_with_center, catch_vtk_errors):
    if _vtk.is_vtk_attribute(class_with_center, 'center'):
        pytest.skip('center is defined by vtk, not pyvista.')
    if class_with_center is pv.DataSetMapper:
        catch_vtk_errors.skip = True

    # Define kwargs as required for some cases.
    kwargs = {}
    if class_with_center is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif class_with_center is pv.Renderer:
        kwargs['parent'] = pv.Plotter()

    instance = try_init_object(class_with_center, kwargs)

    # Test type at runtime
    center = instance.center
    assert len(center) == 3
    assert isinstance(center, tuple)
    assert is_all_floats(center)

    # Test type annotations
    return_type = get_property_return_type(class_with_center.center)
    assert return_type == 'tuple[float, float, float]'
