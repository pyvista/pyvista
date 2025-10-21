"""Module responsible to build the cyclopts App object."""

from __future__ import annotations

from cyclopts import App
from rich.console import Console

import pyvista

app = App(
    name=pyvista.__name__,
    help_format='md',
    version=f'{pyvista.__name__} {pyvista.__version__}',
    help_on_error=True,
    console=Console(),
    result_action='return_value',
    help_formatter='default',
)
