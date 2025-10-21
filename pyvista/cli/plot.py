"""`pyvista plot file.vtp --color=red ...` CLI."""

from __future__ import annotations

from ast import literal_eval
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal

from cyclopts import Parameter
from cyclopts import Token
from rich.console import Group
from rich.console import NewLine
from rich.panel import Panel
from rich.text import Text

import pyvista
from pyvista.core.utilities.misc import StrEnum  # type: ignore [attr-defined]

from .app import app
from .utils import HELP_FORMATTER
from .utils import _console_error
from .utils import _converter_files
from .utils import _MeshAndPath

if TYPE_CHECKING:
    from collections.abc import Sequence


def _validator_window_size(type_: type, value: list[int] | None) -> None:  # noqa: ARG001
    if value is not None and len(value) != 2:
        msg = 'Window size must be a list of two integers.'
        raise ValueError(msg)


def _kwargs_converter(type_, tokens: Sequence[Token]):  # noqa: ANN001, ANN202, ARG001
    for token in tokens:
        # Check hyphen in keyword value
        if (h := '-') in (key := token.keys[0]):
            msg = f'A hyphen `{h}` has been used as supplementary keyword argument and is not converted to underscore `_`. Did you mean --{key.replace("-", "_")}={token.value} ?'  # noqa: E501
            app.console.print(
                Panel(
                    msg,
                    style='magenta',
                    title='Warning',
                    title_align='left',
                )
            )

        # Coerce using literal_eval with fallback to str value
        try:
            return literal_eval(token.value)
        except (ValueError, SyntaxError):
            return token.value
    return None


_HELP_KWARGS = """\
Additional keyword arguments passed to ``Plotter.add_mesh`` or ``Plotter.add_volume``.
See the documentation for more details at https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh
and https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_volume

Note that contrary to other CLI arguments, hyphens ``-`` are not converted to underscores ``_``
before being passed to the corresponding plotter method. For example, you need to use
``--show_edges=True`` instead of ``--show-edges=True`` to show mesh edges in the plotting window.

"""


class Groups(StrEnum):
    """Groups for plot CLI arguments."""

    PLOTTER = 'Plotter init'
    RENDERING = 'Rendering'
    SUPP = 'Supplementary'
    IN = 'Inputs'
    RETURN = 'Return'


@app.command(
    usage=f'Usage: [bold]{pyvista.__name__} plot file (file2) [OPTIONS]',
    help_formatter=HELP_FORMATTER,
    help='Plot one or more mesh files in an interactive window that can be customized with various options.',  # noqa: E501
)
def _plot(
    var_item: Annotated[
        list[str],
        Parameter(
            name='files',
            consume_multiple=True,
            help='File(s) to plot. Must be readable with ``pyvista.read``.',
            converter=_converter_files,
            group=Groups.IN,
            negative='',
        ),
    ],
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
    items: list[_MeshAndPath] = var_item  # type: ignore [assignment]
    try:
        res = pyvista.plot(
            var_item=[m.mesh for m in items],  # type: ignore [arg-type]
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

    except Exception as ex:  # noqa: BLE001
        # Prevent traceback and output error along with help message
        app.help_print(tokens='plot')

        msg = Group(
            ':warning: The following exception has been raised when calling [u]pv.plot[/u]:',
            NewLine(),
            Panel(
                str(ex), title=f'{type(ex).__name__}', title_align='left', style='bold blink red'
            ),
            NewLine(),
            Text('Please check the provided arguments.'),
        )
        _console_error(app=app, message=msg)
    else:
        return res


_plot.__doc__ = pyvista.plot.__doc__  # Needed by cyclopts to get parameters help
