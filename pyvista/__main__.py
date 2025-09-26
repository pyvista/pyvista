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
COMMANDS_DISPLAY = {'report': 'pyvista.Report()'}
COMMANDS_URL = {'report': 'https://docs.pyvista.org/api/utilities/_autosummary/pyvista.report'}


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

        # Convert bool-like strings into literal Python bools
        lower = value.lower()
        if lower in ['true', 'y', 'yes']:
            value = 'True'
        elif lower in ['false', 'n', 'no']:
            value = 'False'

        try:
            # Try Python literal first
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback: treat as string
            kwargs[key] = value
    return kwargs


def main(argv: list[str] | None = None) -> None:
    """PyVista Command-Line Interface entry point."""
    # Use sys.argv if no arguments were explicitly passed
    if argv is None:
        argv = sys.argv[1:]

    # Create top-level parser with --version option
    parser = argparse.ArgumentParser(prog='pyvista')
    parser.add_argument(
        '--version',
        action='version',
        version=f'PyVista {pyvista.__version__}',
        help='show PyVista version and exit',
    )

    # Create a generic keyword subparser for each command
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    for name in COMMANDS:
        url = COMMANDS_URL[name]
        display = COMMANDS_DISPLAY[name]
        subparser = subparsers.add_parser(
            name,
            help=f'run {display!r} with optional key=value kwargs',
            usage='%(prog)s [key=value] ...',
            epilog=f'See documentation for available keywords and more info:\n{url}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparser.add_argument(
            'kwargs', nargs='*', help='optional keyword arguments in key=value form'
        )

    # Parse and validate primary args
    if not argv:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)

    subcommand = args.subcommand
    if subcommand not in COMMANDS:
        parser.print_help()
        sys.exit(1)

    # Parse remaining args as a Python kwargs dict and execute command as a function
    kwargs = _parse_kwargs(getattr(args, 'kwargs', []))
    func = COMMANDS[subcommand]
    result = func(**kwargs)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
