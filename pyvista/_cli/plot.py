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
from .utils import Groups
from .utils import _kwargs_converter
from .utils import border
from .utils import border_color
from .utils import border_width
from .utils import call_or_exit
from .utils import cpos
from .utils import full_screen
from .utils import interactive
from .utils import off_screen
from .utils import read_meshes
from .utils import screenshot
from .utils import show_axes
from .utils import show_bounds
from .utils import skip_unreadable
from .utils import window_size
from .utils import zoom


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
    off_screen: off_screen = None,
    full_screen: full_screen = None,
    screenshot: screenshot = None,
    interactive: interactive = True,
    window_size: window_size = None,
    cpos: cpos = None,
    show_bounds: show_bounds = False,
    show_axes: show_axes = None,
    background: Annotated[str | None, Parameter(group=Groups.RENDERING)] = None,
    text: Annotated[str, Parameter(group=Groups.RENDERING)] = '',
    eye_dome_lighting: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    volume: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    parallel_projection: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    return_cpos: Annotated[bool, Parameter(group=Groups.RETURN)] = False,
    anti_aliasing: Annotated[
        Literal['ssaa', 'msaa', 'fxaa'] | None, Parameter(group=Groups.RENDERING)
    ] = None,
    zoom: zoom = None,
    border: border = None,
    border_color: border_color = None,
    border_width: border_width = 2.0,
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
