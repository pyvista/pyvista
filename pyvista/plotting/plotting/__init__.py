"""Deprecated pyvista.plotting.plotting module."""

from __future__ import annotations

import importlib
import inspect
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning


def __getattr__(name):
    module = importlib.import_module('pyvista.plotting.plotter')
    try:
        value = inspect.getattr_static(module, name)
    except AttributeError:
        module = importlib.import_module('pyvista.plotting')
        try:
            value = inspect.getattr_static(module, name)
        except AttributeError:
            raise AttributeError(
                f'Module `pyvista.plotting.plotting` has been deprecated and we could not automatically find `{name}`.',
            ) from None
    import_path = f'from pyvista.plotting import {name}'
    message = f'The `pyvista.plotting.plotting` module has been deprecated. `{name}` is now imported as: `{import_path}`.'
    warnings.warn(
        message,
        PyVistaDeprecationWarning,
    )
    return value
