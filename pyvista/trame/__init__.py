"""Deprecated shim — use :mod:`trame_pyvista` instead.

The Trame integration moved to the standalone ``trame-pyvista`` package
in PyVista 0.49. Importing from ``pyvista.trame`` is deprecated and
will be removed in a future release.
"""

from __future__ import annotations

from trame_pyvista import PyVistaLocalView
from trame_pyvista import PyVistaRemoteLocalView
from trame_pyvista import PyVistaRemoteView
from trame_pyvista import elegantly_launch
from trame_pyvista import launch_server
from trame_pyvista import show_trame
from trame_pyvista.ui import get_viewer
from trame_pyvista.ui import plotter_ui

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

warn_external(
    'pyvista.trame is deprecated; import from trame_pyvista instead. '
    'Install via `pip install trame-pyvista`.',
    PyVistaDeprecationWarning,
)

__all__ = [
    'PyVistaLocalView',
    'PyVistaRemoteLocalView',
    'PyVistaRemoteView',
    'elegantly_launch',
    'get_viewer',
    'launch_server',
    'plotter_ui',
    'show_trame',
]
