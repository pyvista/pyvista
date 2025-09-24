"""Deprecated utilities subpackage."""

from __future__ import annotations

import importlib
import inspect
import warnings

# Places to look for the utility
_MODULES = [
    'pyvista.core.errors',
    'pyvista.plotting.errors',
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


def __getattr__(name):
    """Fetch an attribute ``name`` from ``globals()`` and warn if it's from a deprecated module.

    Note that ``__getattr__()`` only gets called when ``name`` is missing
    from the module's globals. The trick is that we want to import this
    function into other deprecated modules, and we want to carry this
    subpackage's globals along to prevent some spurious warnings.

    Raises
    ------
    AttributeError
        If the attribute is not found in ``globals()`` and also could not be
        imported from the modules in ``_MODULES``.

    Warns
    -----
    PyVistaDeprecationWarning
        If the attribute has been found via importing from the modules in
        ``_MODULES``, as this implies that the feature has been moved from
        ``pyvista.utilities``.

    """
    from pyvista.core.errors import PyVistaDeprecationWarning  # noqa: PLC0415

    for module in _MODULES:
        feature, import_path = _try_import(module, name)
        if feature is not None:
            break
    else:  # pragma: no cover
        msg = (
            f'Module `pyvista.errors` has been deprecated and we could not automatically '
            f'find `{name}`. This feature has moved.'
        )
        raise AttributeError(msg) from None

    message = (
        f'The `pyvista.errors` module has been deprecated. '
        f'`{name}` is now imported as: `{import_path}`.'
    )

    warnings.warn(
        message,
        PyVistaDeprecationWarning,
        stacklevel=2,
    )

    return feature
