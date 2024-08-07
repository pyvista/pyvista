from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista.core.errors import VTKVersionError

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Iterable


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


def test_bounds_tuple(class_with_bounds):
    # Define kwargs as required for some cases.
    kwargs = {}
    if class_with_bounds is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif class_with_bounds is pv.Renderer:
        kwargs['parent'] = pv.Plotter()

    instance = try_init_object(class_with_bounds, kwargs)

    # Do test
    bounds = instance.bounds
    assert len(bounds) == 6
    assert isinstance(bounds, pv.BoundsTuple)
    assert is_all_floats(bounds)
