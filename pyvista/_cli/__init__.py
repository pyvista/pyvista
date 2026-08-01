"""Command line initialization.

The imports are needed to initialize the various subcommands (ie. plot, report, convert ...)
"""

# Not sorting imports because it impacts the order commands appear in the `--help` page output.
# ruff: noqa: I001
from __future__ import annotations

from .app import CLI_APP as CLI_APP
from . import report as report
from . import convert as convert
from . import plot as plot
from . import validate as validate
