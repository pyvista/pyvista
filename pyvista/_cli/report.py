"""`pyvista report` CLI."""

from __future__ import annotations

from functools import wraps
from typing import get_type_hints

import pyvista as pv
from pyvista import Report

from .app import app
from .utils import HELP_FORMATTER

# Assign annotations to be able to use the Report class using
# cyclopts when __future__ annotations are enabled. See https://github.com/BrianPugh/cyclopts/issues/570
Report.__init__.__annotations__ = get_type_hints(Report.__init__)


@app.command(
    usage=f'Usage: [bold]{pv.__name__} report [ARGS]',
    help_formatter=HELP_FORMATTER,
    help='Generate a PyVista software environment report.',
)
@wraps(Report)
def _report(*args, **kwargs) -> Report:
    return Report(*args, **kwargs)
