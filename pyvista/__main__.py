"""PyVista's command-line interface."""

from __future__ import annotations

import argparse
import ast
import sys
from typing import TYPE_CHECKING
from typing import Any

import pyvista

if TYPE_CHECKING:
    from collections.abc import Callable

COMMANDS: dict[str, Callable[..., Any]] = {
    '--version': pyvista.__version__,  # help='Show PyVista version and exit.'
    'report': pyvista.Report,
}


def _parse_kwargs(args: list[str]) -> dict[str, Any]:
    """Parse CLI args of form key=value into a dict.

    Try literal_eval first, fallback to string.
    """
    kwargs = {}
    for arg in args:
        if '=' not in arg:
            msg = f'Invalid kwarg format: {arg!r}, expected key=value'
            raise ValueError(msg)
        key, value = arg.split('=', 1)
        try:
            # Try Python literal first
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback: treat as string
            kwargs[key] = value
    return kwargs


def main(argv: list[str] | None = None) -> None:
    """PyVista Command-Line Interface entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='PyVista')

    # Create a generic subparser for each command
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    for name in COMMANDS:
        subparsers.add_parser(name, help=f'Run PyVista {name} with kwargs')

    if not argv:
        parser.print_help()
        sys.exit(1)

    subcommand = argv[0]
    if subcommand not in COMMANDS:
        parser.print_help()
        sys.exit(1)

    kwargs = _parse_kwargs(argv[1:])
    func = COMMANDS[subcommand]
    result = func(**kwargs)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
