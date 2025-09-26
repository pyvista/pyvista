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
    'report': pyvista.Report,
}
COMMANDS_DISPLAY = {pyvista.Report: 'pyvista.Report()'}


def _parse_kwargs(args: list[str]) -> dict[str, Any]:
    """Parse CLI args of form key=value into a dict.

    Try literal_eval first, fallback to string.
    """
    kwargs = {}
    for arg in args:
        if arg in ('-h', '--help'):
            # let argparse handle this
            continue
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

    parser = argparse.ArgumentParser(prog='pyvista')
    parser.add_argument(
        '--version',
        action='version',
        version=f'PyVista {pyvista.__version__}',
        help='show PyVista version and exit',
    )

    # Create a generic subparser for each command
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    for name in COMMANDS:
        subparser = subparsers.add_parser(
            name,
            help=f'run {COMMANDS_DISPLAY[COMMANDS[name]]!r} with optional key=value kwargs',
            usage='%(prog)s [key=value] ...',
        )
        subparser.add_argument(
            'kwargs', nargs='*', help='optional keyword arguments in key=value form'
        )
    if not argv:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    subcommand = args.subcommand
    if subcommand not in COMMANDS:
        parser.print_help()
        sys.exit(1)

    kwargs = _parse_kwargs(getattr(args, 'kwargs', []))
    func = COMMANDS[subcommand]
    result = func(**kwargs)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
