"""Deprecated pyvista.plotting.plotting module."""
import importlib
import inspect
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning


class PyVistaPlottingDeprecationWarning(PyVistaDeprecationWarning):
    """Deprecation warning specific to `pyvista.plotting.plotting`."""

    pass


def __getattr__(name):
    module = importlib.import_module('pyvista.plotting.plotter')
    try:
        value = inspect.getattr_static(module, name)
        import_path = f'from pyvista.plotting.plotter import {name}'
    except AttributeError:
        module = importlib.import_module('pyvista.plotting')
        try:
            value = inspect.getattr_static(module, name)
            import_path = f'from pyvista.plotting import {name}'
        except AttributeError:
            raise AttributeError(
                f'Module `pyvista.plotting.plotting` has been deprecated and we could not automatically find `{name}`.'
            )

    message = f'The `pyvista.plotting.plotting` module has been deprecated. `{name}` is now imported as: `{import_path}`.'

    warnings.warn(
        message,
        PyVistaPlottingDeprecationWarning,
    )
    return value
