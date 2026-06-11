"""Deprecated shim — use :mod:`trame_pyvista.widgets` instead."""

from __future__ import annotations

from trame_pyvista.widgets import CLOSED_PLOTTER_ERROR
from trame_pyvista.widgets import PyVistaLocalView
from trame_pyvista.widgets import PyVistaRemoteLocalView
from trame_pyvista.widgets import PyVistaRemoteView
from trame_pyvista.widgets import _BasePyVistaView
from trame_pyvista.widgets import get_server

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

warn_external(
    'pyvista.trame.views is deprecated; import from trame_pyvista.widgets instead.',
    PyVistaDeprecationWarning,
)

__all__ = [
    'CLOSED_PLOTTER_ERROR',
    'PyVistaLocalView',
    'PyVistaRemoteLocalView',
    'PyVistaRemoteView',
    '_BasePyVistaView',
    'get_server',
]
