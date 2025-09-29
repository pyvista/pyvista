"""PyVista's command-line interface."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

from cyclopts import App

import pyvista
from pyvista import Report

if TYPE_CHECKING:
    from collections.abc import Callable


def _report(**kwargs):  # noqa: ANN202
    return Report(**kwargs)


COMMANDS: dict[str, Callable[..., Any]] = {
    'report': _report,
    'plot': pyvista.plot,
}
COMMANDS_DISPLAY = {
    'report': 'pyvista.Report()',
    'plot': 'pyvista.plot()',
}
COMMANDS_URL = {
    'report': 'https://docs.pyvista.org/api/utilities/_autosummary/pyvista.report',
    'plot': 'https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plot.html',
}


app = App(
    version=f'{pyvista.__name__} {pyvista.__version__}',
    help_format='plaintext',
)


for key, val in COMMANDS.items():
    url = COMMANDS_URL[key]
    display = COMMANDS_DISPLAY[key]

    app.command(
        val,
        help_format='plaintext',
        help=f'See documentation for available arguments and keywords:\n{url}',
    )


def main(argv: list[str] | None = None) -> None:
    """PyVista Command-Line Interface entry point."""
    # Use sys.argv if no arguments were explicitly passed
    if argv is None:
        argv = sys.argv[1:]

    result = app(tokens=argv)
    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
