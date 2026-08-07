"""`pyvista plot file.vtp --color=red ...` CLI."""

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Literal

from cyclopts import Parameter

import pyvista as pv

from .app import CLI_APP
from .utils import HELP_FORMATTER
from .utils import HELP_KWARGS
from .utils import CposView
from .utils import Groups
from .utils import _kwargs_converter
from .utils import _validator_window_size
from .utils import call_or_exit
from .utils import read_meshes
from .utils import skip_unreadable


@CLI_APP.command(
    usage=f'Usage: [bold]{pv.__name__} plot PATH... [OPTIONS]',
    help_formatter=HELP_FORMATTER,
    sort_key=0,
)
def _plot(
    paths: Annotated[
        list[str],
        Parameter(
            consume_multiple=True,
            help=(
                'Paths(s) to plot. Glob patterns (``*``, ``?``, ``[...]``) are expanded. '
                'Each match must be readable with ``pyvista.read``.'
            ),
            group=Groups.IN,
        ),
    ],
    /,
    *,
    skip_unreadable: skip_unreadable = False,
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
    cpos: Annotated[CposView | None, Parameter(group=Groups.RENDERING)] = None,
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
    border_color: Annotated[str | None, Parameter(group=Groups.PLOTTER)] = None,
    border_width: Annotated[float | None, Parameter(group=Groups.PLOTTER)] = None,
    ssao: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    static: Annotated[bool, Parameter(group=Groups.SUPP)] = False,
    **kwargs: Annotated[
        Any,
        Parameter(help=HELP_KWARGS, converter=_kwargs_converter, group=Groups.SUPP),
    ],
) -> None:
    """Plot one or more mesh files in an interactive window."""
    meshes = read_meshes(paths, skip_unreadable=skip_unreadable)
    if static:
        kwargs['static'] = True
    return call_or_exit(
        pv.plot,
        command='plot',
        var_item=meshes,
        off_screen=off_screen,
        full_screen=full_screen,
        screenshot=screenshot,
        interactive=interactive,
        cpos=cpos,
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
