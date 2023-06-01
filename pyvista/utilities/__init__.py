"""Deprecated utilities module."""
import importlib
import inspect
import warnings

from pyvista.core.errors import DeprecationError, PyVistaDeprecationWarning


def __dir__():
    raise DeprecationError('The `pyvista.utilities` module has been deprecated.')


def __getattr__(name):
    # TODO: for some reason this runs twice
    utils = importlib.import_module('pyvista.core.utilities')
    try:
        value = inspect.getattr_static(utils, name)
        import_path = f'from pyvista.core.utilities import {name}'
    except AttributeError:
        utils = importlib.import_module('pyvista.plotting.utilities')
        try:
            value = inspect.getattr_static(utils, name)
            import_path = f'from pyvista.plotting.utilities import {name}'
        except AttributeError:
            raise AttributeError(
                f'Module `pyvista.utilities` has been deprecated and we could not automatically find `{name}` in `pyvista.core.utilities` or `pyvista.plotting.utilities`.'
            )

    message = f'The `pyvista.utilities` module has been deprecated. `{name}` is now imported as: `{import_path}`.'
    if inspect.ismodule(value):
        message += (
            f' `{name}` is an internal module and its members may have changed during the refactor.'
        )

    warnings.warn(
        message,
        PyVistaDeprecationWarning,
    )
    return value
