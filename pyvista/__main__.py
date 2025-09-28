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


def _parse_args_and_kwargs(args: list[str]) -> tuple[list[Any], dict[str, Any]]:
    """Parse CLI args into a list of python args and key=value pairs into a dict."""

    def _literal_eval(val: str) -> Any:
        """Evaluate as python literals."""
        try:
            # Try Python literal first
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # Fallback: treat as string
            return val

    py_args = []
    py_kwargs = {}
    positional = True
    for arg in args:
        if not '=' in arg:
            # Expect positonal py arg
            if not positional:
                msg = f'Positional argument {arg} must not follow a keyword argument.'
                raise SyntaxError(msg)
            py_args.append(_literal_eval(arg))
            continue

        # Expect keywords only for the remainder of parsing
        positional = False

        key, value = arg.split('=', 1)

        # Convert bool-like strings into literal Python bools
        lower = value.lower()
        if lower in ['true', 'y', 'yes']:
            value = 'True'
        elif lower in ['false', 'n', 'no']:
            value = 'False'

        py_kwargs[key] = _literal_eval(value)

    return py_args, py_kwargs


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
        version=f'pyvista {pyvista.__version__}',
        help='show PyVista version and exit',
    )

    # Create a generic keyword subparser for each command
    subparsers = parser.add_subparsers(dest='subcommand', required=True)
    for name in COMMANDS:
        url = COMMANDS_URL[name]
        display = COMMANDS_DISPLAY[name]
        subparser = subparsers.add_parser(
            name,
            help=f'run {display!r} with python args and key=value kwargs',
            usage='%(prog)s [args ...] [key=value ...]',
            epilog=f'See documentation for available arguments and keywords:\n{url}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparser.add_argument(
            'args', nargs='*', help='required and/or optional python positional arguments'
        )
        subparser.add_argument(
            'kwargs', nargs='*', help='optional python keyword arguments in key=value form'
        )

    # Parse primary args
    if not argv:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(argv)

    # Parse remaining args as a Python kwargs dict and execute command as a function
    py_args, py_kwargs = _parse_args_and_kwargs(getattr(args, 'args', []))
    func = COMMANDS[args.subcommand]
    result = func(*py_args, **py_kwargs)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
