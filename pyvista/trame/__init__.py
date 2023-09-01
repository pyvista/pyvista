"""Trame interface for PyVista."""
import logging

logging.getLogger('trame.app').disabled = True

from pyvista.trame.jupyter import elegantly_launch, launch_server, show_trame
from pyvista.trame.ui import get_viewer, plotter_ui
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView

# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'elegantly_launch',
    'get_viewer',
    'launch_server',
    'plotter_ui',
    'show_trame',
    'PyVistaLocalView',
    'PyVistaRemoteLocalView',
    'PyVistaRemoteView',
]
