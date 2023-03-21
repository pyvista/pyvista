"""Trame interface for PyVista."""
from pyvista.trame.jupyter import launch_server, show_trame, elegantly_launch
from pyvista.trame.ui import get_or_create_viewer, plotter_ui
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView


# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'elegantly_launch',
    'get_or_create_viewer',
    'launch_server',
    'plotter_ui',
    'show_trame',
    'PyVistaLocalView',
    'PyVistaRemoteLocalView',
    'PyVistaRemoteView',
]
