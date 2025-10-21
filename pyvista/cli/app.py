"""Module responsible to build the cyclopts App object."""

from __future__ import annotations

from cyclopts import App
from rich.console import Console

import pyvista

# help format needs to be `plaintext` due to unsupported `versionadded` sphinx directive in rich.
# Change to `rst` when cyclopts v4 is out. See https://github.com/BrianPugh/cyclopts/issues/568
app = App(
    name=pyvista.__name__,
    help_format='plaintext',
    version=f'{pyvista.__name__} {pyvista.__version__}',
    help_on_error=True,
    console=Console(),
    result_action='return_value',
)
