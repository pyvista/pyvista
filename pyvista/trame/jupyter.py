"""Deprecated shim — use :mod:`trame_pyvista.jupyter` instead."""

from __future__ import annotations

from trame_pyvista.jupyter import EmbeddableWidget
from trame_pyvista.jupyter import TrameJupyterServerDownError
from trame_pyvista.jupyter import TrameServerDownError
from trame_pyvista.jupyter import Widget
from trame_pyvista.jupyter import build_url
from trame_pyvista.jupyter import elegantly_launch
from trame_pyvista.jupyter import initialize
from trame_pyvista.jupyter import launch_server
from trame_pyvista.jupyter import show_trame

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

warn_external(
    'pyvista.trame.jupyter is deprecated; import from trame_pyvista.jupyter instead.',
    PyVistaDeprecationWarning,
)

__all__ = [
    'EmbeddableWidget',
    'TrameJupyterServerDownError',
    'TrameServerDownError',
    'Widget',
    'build_url',
    'elegantly_launch',
    'initialize',
    'launch_server',
    'show_trame',
]
