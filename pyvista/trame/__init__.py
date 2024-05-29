"""Trame interface for PyVista."""

from __future__ import annotations

import logging

logging.getLogger('trame.app').disabled = True

from pyvista.trame.jupyter import elegantly_launch
from pyvista.trame.jupyter import launch_server
from pyvista.trame.jupyter import show_trame
from pyvista.trame.ui import get_viewer
from pyvista.trame.ui import plotter_ui
from pyvista.trame.views import PyVistaLocalView
from pyvista.trame.views import PyVistaRemoteLocalView
from pyvista.trame.views import PyVistaRemoteView

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
