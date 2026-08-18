"""Deprecated shim — use :mod:`trame_pyvista.ui.vuetify3` instead."""

from __future__ import annotations

from trame_pyvista.ui.vuetify3 import Viewer
from trame_pyvista.ui.vuetify3 import button
from trame_pyvista.ui.vuetify3 import checkbox
from trame_pyvista.ui.vuetify3 import divider
from trame_pyvista.ui.vuetify3 import select
from trame_pyvista.ui.vuetify3 import slider
from trame_pyvista.ui.vuetify3 import text_field

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning

warn_external(
    'pyvista.trame.ui.vuetify3 is deprecated; import from trame_pyvista.ui.vuetify3 instead.',
    PyVistaDeprecationWarning,
)

__all__ = [
    'Viewer',
    'button',
    'checkbox',
    'divider',
    'select',
    'slider',
    'text_field',
]
