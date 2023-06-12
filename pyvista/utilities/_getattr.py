import importlib
import inspect
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning

# Places to look for the utility
_MODULES = [
    'pyvista.core.utilities.cells',
    'pyvista.core.utilities',
    'pyvista.core.errors',
    'pyvista.core',
    'pyvista.plotting.utilities',
    'pyvista.plotting.errors',
    'pyvista.plotting.texture',
    'pyvista.plotting',
    'pyvista.report',
    'pyvista.themes',
]


def _try_import(module, name):
    """Attempt to import a module."""
    _module = importlib.import_module(module)
    try:
        feature = inspect.getattr_static(_module, name)
        import_path = f'from {module} import {name}'
    except AttributeError:
        return None, None
    return feature, import_path


def _getattr_factory(globs):
    """Create and return a custom getattr method.

    The created getattr method tries to dynamically fetch an attribute ``name``
    from the given dictionary globs, expected to be ``globals``.

    Raises
    ------
    AttributeError
        If the attribute is not found in ``globs`` and also could not be
        imported from the modules in _MODULES.

    PyVistaDeprecationWarning
        If the attribute has been found via importing from the modules in
        ``_MODULES``, as this implies that the feature has been moved from
        pyvista.utilities.

    Notes
    -----
    This function is intended to be used internally and is part of handling
    backward compatibility for transition from deprecated modules.

    """

    def __getattr__(name):
        try:
            return globs[name]
        except KeyError:
            pass

        for module in _MODULES:
            feature, import_path = _try_import(module, name)
            if feature is not None:
                break
        else:
            raise AttributeError(
                f'Module `pyvista.utilities` has been deprecated and we could not automatically find `{name}`. This feature has moved.'
            ) from None

        message = f'The `pyvista.utilities` module has been deprecated. `{name}` is now imported as: `{import_path}`.'

        warnings.warn(
            message,
            PyVistaDeprecationWarning,
        )

        return feature

    return __getattr__
