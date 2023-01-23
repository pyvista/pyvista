"""Trame interface for PyVista."""
from pyvista.trame.jupyter import show_trame
from pyvista.trame.ui import initialize, plotter_ui
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView


# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'initialize',
    'plotter_ui',
    'show_trame',
    'PyVistaLocalView',
    'PyVistaRemoteLocalView',
    'PyVistaRemoteView',
]
