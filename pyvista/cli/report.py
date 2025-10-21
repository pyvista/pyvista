"""`pyvista report` CLI."""

from __future__ import annotations

from functools import wraps
from typing import get_type_hints

from pyvista import Report

from .app import app
from .utils import HELP_FORMATTER

# Assign annotations to be able to use the Report class using
# cyclopts when __future__ annotations are enabled. See https://github.com/BrianPugh/cyclopts/issues/570
Report.__init__.__annotations__ = get_type_hints(Report.__init__)


@app.command(help_formatter=HELP_FORMATTER)
@wraps(Report)
def _report(*args, **kwargs) -> Report:
    return Report(*args, **kwargs)
