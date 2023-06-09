import importlib
import inspect
import pathlib
import warnings

# Have to set __path__ to this file's directory so that
# `import pyvista.utilities` imports this directory
__path__ = [str(pathlib.Path(__file__).parent)]

# Places to look for the utility
_MODULES = [
    'pyvista.core.utilities',
    'pyvista.core.errors',
    'pyvista.core',
    'pyvista.plotting.utilities',
    'pyvista.plotting.errors',
    'pyvista.plotting.texture',
    'pyvista.report',
    'pyvista.themes',
]


def _try_import(module, name):
    _module = importlib.import_module(module)
    try:
        feature = inspect.getattr_static(_module, name)
        import_path = f'from {module} import {name}'
    except AttributeError:
        return None, None
    return feature, import_path


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass

    from pyvista.core.errors import PyVistaDeprecationWarning

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
