"""Command line initialization."""

# Sort imports manually
# The import order is used to show the possible subcommands in the error message when an unknown
# command is encountered. This order should be the same as the order dictated by the ``sort_key``
# keyword used for each subcommand.
# ruff: noqa: I001
from __future__ import annotations

from .app import CLI_APP as CLI_APP
from . import plot as plot
from . import convert as convert
from . import validate as validate
from . import report as report
