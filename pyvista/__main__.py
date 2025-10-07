"""PyVista's command-line interface."""

from __future__ import annotations

from ast import literal_eval
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import get_type_hints
import warnings

from cyclopts import App
from cyclopts import Parameter
from cyclopts import Token
from rich import box
from rich.console import Group
from rich.console import NewLine
from rich.panel import Panel
from rich.text import Text

import pyvista
from pyvista import Report
from pyvista.core.errors import PyVistaDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

# Assign annotations to be able to use the Report class using
# cyclopts when __future__ annotations are enabled. See https://github.com/BrianPugh/cyclopts/issues/570
Report.__init__.__annotations__ = get_type_hints(Report.__init__)


# help format needs to be `plaintext` due to unsupported `versionadded` sphinx directive in rich.
# Change to `rst` when cyclopts v4 is out. See https://github.com/BrianPugh/cyclopts/issues/568
app = App(
    name=pyvista.__name__,
    help_format='plaintext',
    version=f'{pyvista.__name__} {pyvista.__version__}',
    help_on_error=True,
)


@app.command
@wraps(Report)
def report(*args, **kwargs):  # noqa: ANN201, D103
    return Report(*args, **kwargs)


def _validator_window_size(type_: type, value: list[int] | None) -> None:  # noqa: ARG001
    if value is not None and len(value) != 2:
        msg = 'Window size must be a list of two integers.'
        raise ValueError(msg)


def _validator_files(type_: type, value: list[str] | None) -> None:  # noqa: ARG001
    if value is None:
        return

    # Test file exists
    if not all((files := {v: Path(v).exists() for v in value}).values()):
        missing = [k for k, v in files.items() if not v]
        msg = f'File(s) not found: {missing}'
        raise ValueError(msg)

    # Test file can be read by pyvista
    def readable(file: str) -> bool:
        try:
            pyvista.read(file)
        except Exception:  # noqa: BLE001
            return False
        else:
            return True

    if not all((files := {v: readable(v) for v in value}).values()):
        not_readable = [k for k, v in files.items() if not v]
        msg = f'File(s) not readable by pyvista: {not_readable}'
        raise ValueError(msg)


def _kwargs_converter(type_, tokens: Sequence[Token]):  # noqa: ANN001, ANN202, ARG001
    for token in tokens:
        # Check hyphen in keyword value
        if (h := '-') in (key := token.keys[0]):
            msg = f'cannot use hyphen `{h}`, try with --{key.replace("-", "_")}={token.value}'
            raise ValueError(msg)

        # Coerce using literal_eval with fallback to str value
        try:
            return literal_eval(token.value)
        except (ValueError, SyntaxError):
            return token.value
    return None


_HELP_KWARGS = """\
Additional keyword arguments passed to ``Plotter.add_mesh``.
See documentation for more details https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#pyvista.Plotter.add_mesh

Note that contrary to other arguments, hyphens CANNOT not be used (ie. use ``--show_edges=True`` instead of ``--show-edges=True``).

"""  # noqa: E501


class Groups(StrEnum):
    """Groups for plot CLI arguments."""

    PLOTTER = 'Plotter init'
    RENDERING = 'Rendering'
    SUPP = 'Supplementary'
    IN = 'Inputs'
    RETURN = 'Return'


@app.command(usage=f'Usage: [bold]{pyvista.__name__} plot file (file2) [OPTIONS]')
def _plot(
    files: Annotated[
        list[str] | None,
        Parameter(
            consume_multiple=True,
            help='File(s) to plot. Must be readable with ``pv.read``. If nothing is provided, show an empty window.',  # noqa: E501
            validator=_validator_files,
            group=Groups.IN,
        ),
    ] = None,
    *,
    off_screen: Annotated[bool | None, Parameter(group=Groups.PLOTTER)] = None,
    full_screen: Annotated[bool | None, Parameter(group=Groups.RENDERING)] = None,
    screenshot: Annotated[str | None, Parameter(group=Groups.PLOTTER)] = None,
    interactive: Annotated[bool, Parameter(group=Groups.PLOTTER)] = True,
    window_size: Annotated[
        list[int] | None,
        Parameter(
            consume_multiple=True,
            validator=_validator_window_size,
            group=Groups.PLOTTER,
        ),
    ] = None,
    show_bounds: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    show_axes: Annotated[bool | None, Parameter(group=Groups.RENDERING)] = None,
    background: Annotated[str | None, Parameter(group=Groups.RENDERING)] = None,
    text: Annotated[str, Parameter(group=Groups.RENDERING)] = '',
    eye_dome_lighting: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    volume: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    parallel_projection: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    return_cpos: Annotated[bool, Parameter(group=Groups.RETURN)] = False,
    anti_aliasing: Annotated[
        Literal['ssaa', 'msaa', 'fxaa'] | None, Parameter(group=Groups.RENDERING)
    ] = None,
    zoom: Annotated[float | str | None, Parameter(group=Groups.RENDERING)] = None,
    border: Annotated[bool, Parameter(group=Groups.PLOTTER)] = False,
    border_color: Annotated[str, Parameter(group=Groups.PLOTTER)] = 'k',
    border_width: Annotated[float, Parameter(group=Groups.PLOTTER)] = 2.0,
    ssao: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    **kwargs: Annotated[
        Any,
        Parameter(help=_HELP_KWARGS, converter=_kwargs_converter, group=Groups.SUPP),
    ],
) -> None:
    try:
        res = pyvista.plot(
            var_item=files or [],  # type: ignore[arg-type]
            off_screen=off_screen,
            full_screen=full_screen,
            screenshot=screenshot,
            interactive=interactive,
            window_size=window_size,
            show_bounds=show_bounds,
            show_axes=show_axes,
            background=background,
            text=text,
            eye_dome_lighting=eye_dome_lighting,
            volume=volume,
            parallel_projection=parallel_projection,
            return_cpos=return_cpos,
            anti_aliasing=anti_aliasing,
            zoom=zoom,
            border=border,
            border_color=border_color,
            border_width=border_width,
            ssao=ssao,
            **kwargs,
        )

    except Exception as ex:
        # Prevent traceback and output error along with help message
        console = app._resolve_console(tokens_or_apps=None)
        app.help_print(tokens='plot', console=console)

        msg = Group(
            ':warning: The following exception has been raised when calling [u]pv.plot[/u]:',
            NewLine(),
            Panel(
                str(ex), title=f'{type(ex).__name__}', title_align='left', style='bold blink red'
            ),
            NewLine(),
            Text('Please check the provided arguments.'),
        )
        panel = Panel(
            msg,
            title='Pyvista error',
            style='red',
            box=box.ROUNDED,
            expand=True,
            title_align='left',
        )

        console.print(panel)
        raise SystemExit(1) from ex
    else:
        return res


_plot.__doc__ = pyvista.plot.__doc__  # Needed by cyclopts to get parameters help


def main(argv: list[str] | str | None = None) -> None:
    """PyVista Command-Line Interface entry point."""
    # Ignore warnings emitted because arguments are passed positionally by the
    # inspect module. See https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.kwargs
    # and https://github.com/BrianPugh/cyclopts/issues/567
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PyVistaDeprecationWarning)
        result = app(tokens=argv)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
