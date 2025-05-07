"""PyVista's command-line interface."""

from __future__ import annotations

import argparse
import inspect
import re

import pyvista as pv


def _parse_numpy_doc(doc: str) -> dict[str, str]:
    """Parse a NumPy-style docstring to get parameter names and help strings."""
    help_map: dict[str, str] = {}  # Store parameter name to description mapping
    if not doc:
        return help_map

    lines = doc.splitlines()
    in_params = False  # Flag to indicate whether we are inside the 'Parameters' section
    current_param = None  # Temporary variable

    # Iterate through lines of the docstring
    for line in lines:
        stripped = line.strip()

        # Detect the start of the 'Parameters' section
        if re.match(r'^Parameters$', stripped):
            in_params = True
            continue

        # Skip the underline section of 'Parameters'
        if in_params and re.match(r'^-{3,}$', stripped):
            continue

        # Detect a parameter line, e.g., 'param_name : type'
        if in_params and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:', stripped):
            current_param = stripped.split(':')[0].strip()  # Extract the parameter name
            help_map[current_param] = ''  # Initialize description storage
        elif in_params and stripped:
            # Append description to the current parameter
            if current_param:
                help_map[current_param] += ' ' + stripped
        elif in_params and not stripped:
            # End of a parameter description
            current_param = None
        elif in_params and not current_param:
            break  # Stop if we've passed the Parameters section

    return {k: v.strip() for k, v in help_map.items()}  # Clean up the whitespace


def _add_report_subparser(subparsers):  # noqa: ANN001, ANN202
    """Add the 'report' subcommand to the argparse parser.

    Arguments are generated dynamically based on the Report class'
    signature and docstring.
    """
    doc_help = _parse_numpy_doc(pv.Report.__doc__)  # type: ignore[arg-type]
    sig = inspect.signature(
        pv.Report.__init__
    )  # Get the function signature of the Report class' __init__ method
    parser = subparsers.add_parser(
        'report', help='Show system information via pyvista.Report'
    )  # Create a parser for the 'report' command

    # Loop through the signature parameters and add them to the argument parser
    for name, param in sig.parameters.items():
        if name == 'self':  # Skip the 'self' parameter for instance methods
            continue

        # Get the default value (if any) and type
        default = param.default if param.default is not inspect.Parameter.empty else None
        param_type = type(default) if default is not None else str

        # Get the help text from the docstring (if available)
        help_text = doc_help.get(name, '')
        arg_name = (
            f'--{name.replace("_", "-")}'  # Convert underscores to hyphens for argument names
        )

        # Handle bool parameters
        if param_type is bool:
            parser.add_argument(
                arg_name,
                type=lambda x: x.lower()
                != 'false',  # Convert 'false' to False, anything else to True
                default=default,
                help=help_text,
            )
        # Handle lists with `nargs` to allow multiple values
        elif param_type is list:
            parser.add_argument(arg_name, nargs='+', default=default, help=help_text)
        else:
            # General case
            parser.add_argument(arg_name, type=param_type, default=default, help=help_text)

    return parser


def main() -> None:
    """Entry point for the pyvista CLI."""
    # Create the main parser
    parser = argparse.ArgumentParser(description='PyVista CLI')
    parser.add_argument('--version', action='store_true', help='Show PyVista version and exit')

    # Subparser for different commands
    subparsers = parser.add_subparsers(dest='command')
    # Add 'report' subcommand dynamically
    _add_report_subparser(subparsers)

    args = parser.parse_args()

    # Handle the --version flag
    if args.version:
        print(pv.__version__)  # Print PyVista version
    elif args.command == 'report':
        # Pass arguments for the 'report' command to the Report class
        kwargs = vars(args).copy()  # Convert the arguments into a dictionary
        kwargs.pop('command', None)  # Remove the 'command' key as it's not a parameter for Report
        kwargs.pop('version', None)  # Remove the 'version' flag
        print(pv.Report(**kwargs))  # Create a Report instance and print it
    else:
        parser.print_help()  # Show help if no valid command is provided


if __name__ == '__main__':
    main()
