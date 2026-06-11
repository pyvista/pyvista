"""Internal helpers shared by PyVista's plugin registries.

The registries (accessor, reader, writer, interactor style, jupyter
backend) all need to record where each registration came from so a user
inspecting :func:`~pyvista.registered_readers` and friends can tell
explicit code apart from entry-point discovery. The helpers here keep
that source-attribution logic in one place.
"""

from __future__ import annotations


def handler_source(handler: object) -> str:
    """Return ``module.qualname`` for *handler* when available.

    Used by every plugin registry to attach a human-readable origin
    string to each explicit registration so that :func:`registered_*`
    introspection can show where a handler came from.

    Parameters
    ----------
    handler : object
        Any callable or class; typically a function, method, or type
        object whose ``__module__`` and ``__qualname__`` attributes are
        meaningful.

    Returns
    -------
    str
        ``f'{handler.__module__}.{handler.__qualname__}'`` when both
        attributes are present, falling back to ``'<unknown>'`` for the
        module piece and ``repr(handler)`` for the qualname piece when
        not.

    """
    module = getattr(handler, '__module__', None) or '<unknown>'
    qualname = getattr(handler, '__qualname__', None) or repr(handler)
    return f'{module}.{qualname}'
