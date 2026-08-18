"""Deprecated shim — use :mod:`trame_pyvista.ui` instead."""

from __future__ import annotations

from trame_pyvista.ui import UI_TITLE
from trame_pyvista.ui import get_viewer
from trame_pyvista.ui import plotter_ui

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

warn_external(
    'pyvista.trame.ui is deprecated; import from trame_pyvista.ui instead.',
    PyVistaDeprecationWarning,
)

__all__ = [
    'UI_TITLE',
    'get_viewer',
    'plotter_ui',
]
