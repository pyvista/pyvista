from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from pyvista.core.errors import PyVistaAttributeError
from pyvista.core.errors import VTKVersionError

if TYPE_CHECKING:
    from types import ModuleType


def get_all_classes() -> tuple[tuple[str, ...], tuple[type, ...]]:
    """Return all classes which inherit from vtk classes."""
    class_types: list[type] = []

    def _append_classes_from_module(module: ModuleType):
        keys = module.__dict__.keys()
        for module_item in keys:
            module_attr = getattr(module, module_item)
            if not hasattr(module_attr, '__mro__'):
                continue  # not a class
            if any(item.__name__.startswith('vtk') for item in module_attr.__mro__):
                class_types.append(module_attr)

    # Get from core and plotting separately since plotting module has a lazy importer
    _append_classes_from_module(pv.core)
    _append_classes_from_module(pv.plotting)

    # Sort and return names and types as separate tuples
    dict_ = {class_.__name__: class_ for class_ in class_types}
    sorted_dict = {key: dict_[key] for key in sorted(dict_.keys())}
    return tuple(sorted_dict.keys()), tuple(sorted_dict.values())


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'vtk_subclass' in metafunc.fixturenames:
        # Generate a separate test for any class that has bounds
        class_names, class_types = get_all_classes()
        metafunc.parametrize('vtk_subclass', class_types, ids=class_names)


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

    instance = try_init_object(vtk_subclass, kwargs)
    vtk_attr_camel_case = 'GetGlobalWarningDisplay'
    vtk_attr_snake_case = 'global_warning_display'

    # Make sure the CamelCase attribute exists and can be accessed
    assert hasattr(instance, vtk_attr_camel_case)

    if pv.vtk_version_info >= (9, 4):
        # Test getting the snake_case equivalent raises error
        match = "The attribute 'global_warning_display' is defined by VTK and is not part of the PyVista API"
        with pytest.raises(PyVistaAttributeError, match=match):
            getattr(instance, vtk_attr_snake_case)
    else:
        assert not hasattr(instance, vtk_attr_snake_case)
