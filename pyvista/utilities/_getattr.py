import importlib
import inspect
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning

# Places to look for the utility
_MODULES = [
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


class _GetAttr:
    def __init__(self):
        pass

    @staticmethod
    def _try_import(module, name):
        _module = importlib.import_module(module)
        try:
            feature = inspect.getattr_static(_module, name)
            import_path = f'from {module} import {name}'
        except AttributeError:
            return None, None
        return feature, import_path

    def __call__(self, name):
        for module in _MODULES:
            feature, import_path = _GetAttr._try_import(module, name)
            if feature is not None:
                break
        else:
            raise AttributeError(
                f'Module `pyvista.utilities` has been deprecated and we could not automatically find `{name}`. This feature has moved.'
            ) from None

        # Ignore __path__ to avoid confusing warnings. See
        # https://github.com/pyvista/pyvista/pull/4507#discussion_r1225972997
        if name == '__path__':
            return feature

        message = f'The `pyvista.utilities` module has been deprecated. `{name}` is now imported as: `{import_path}`.'

        warnings.warn(
            message,
            PyVistaDeprecationWarning,
        )

        return feature
