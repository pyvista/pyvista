"""`pyvista compare file1.vtp file2.vtp --link=False ...` CLI."""

from __future__ import annotations

import re
from typing import Annotated
from typing import Any
import warnings

from cyclopts import Parameter
from rich.panel import Panel

import pyvista as pv

from .app import CLI_APP
from .utils import HELP_FORMATTER
from .utils import HELP_KWARGS
from .utils import CposView
from .utils import Groups
from .utils import _kwargs_converter
from .utils import _validator_window_size
from .utils import call_or_exit
from .utils import print_error_and_exit
from .utils import read_meshes
from .utils import skip_unreadable
from .utils import validate_paths

_HELP_SHAPE = """\
Shape of the subplot grid, as either the number of rows and columns, e.g. ``2,2``,
or one of the string descriptors accepted by ``Plotter``, e.g. ``3|1`` for three
subplots on the left and one on the right. By default, a compact grid which is
never taller than it is wide is used.
"""

_HELP_LABELS = """\
Labels to show in each subplot. Must be given once per path. By default, the file
name of each path is used.
"""

_HELP_LINK = """\
Share a single camera between the subplots, so that the meshes are shown at a common
scale. By default, the cameras are shared only when every mesh is at least half the
size of all of them together.
"""

_HELP_OUTLINE = """\
Draw an outline of the bounds of every mesh in each subplot, to give the comparison a
common frame of reference.
"""

_HELP_LABEL_SIZE = """\
Font size of the label shown in each subplot. Useful when the file names are long
enough to be cut off.
"""

# What to do about a dataset which is too small to make out, in terms of the options
# this command has rather than the arguments of `plot_compare`, which it does not
_REMEDIES = {
    'reference mesh': 'Omit `--outline` to fit each subplot to its own mesh.',
    'shared camera': 'Use `--no-link` to fit each subplot to its own mesh.',
}


def _report_warnings(caught: list[warnings.WarningMessage]) -> None:
    """Print the warnings raised while plotting, with advice for a command line."""
    for warning in caught:
        message = str(warning.message)
        problem, made_out, _ = message.partition('make out.')
        remedy = next((advice for cause, advice in _REMEDIES.items() if cause in message), None)
        # This command draws the reference mesh itself, so name it as it is spelled here
        problem = problem.replace('the reference mesh', 'the outline')
        if not made_out or remedy is None:
            # Not a warning this command has anything better to say about
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
            continue
        CLI_APP.error_console.print(
            Panel(
                f'{problem}{made_out} {remedy}',
                style='magenta',
                title='Warning',
                title_align='left',
            )
        )


def _parse_shape(shape: str) -> list[int] | str:
    """Return a ``Plotter`` shape from its command line spelling.

    The string descriptors are passed through as they are. Anything else is read as
    the number of rows and columns, which are separated by a comma or a space on the
    command line rather than by the brackets used in Python.
    """
    if any(separator in shape for separator in '|/'):
        return shape
    return [int(value) for value in re.split(r'[,\s]+', shape.strip()) if value]


@CLI_APP.command(
    usage=f'Usage: [bold]{pv.__name__} compare PATH... [OPTIONS]',
    help_formatter=HELP_FORMATTER,
)
def _compare(
    paths: Annotated[
        list[str],
        Parameter(
            consume_multiple=True,
            help=(
                'Path(s) to compare. Glob patterns (``*``, ``?``, ``[...]``) are expanded. '
                'Each match must be readable with ``pyvista.read``. At least two paths are '
                'needed, and each is rendered in its own subplot.'
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
    shape: Annotated[str | None, Parameter(help=_HELP_SHAPE, group=Groups.PLOTTER)] = None,
    labels: Annotated[
        list[str] | None,
        Parameter(consume_multiple=True, help=_HELP_LABELS, group=Groups.RENDERING),
    ] = None,
    link: Annotated[bool | None, Parameter(help=_HELP_LINK, group=Groups.RENDERING)] = None,
    cpos: Annotated[CposView | None, Parameter(group=Groups.RENDERING)] = None,
    outline: Annotated[bool, Parameter(help=_HELP_OUTLINE, group=Groups.RENDERING)] = False,
    label_size: Annotated[
        int | None, Parameter(help=_HELP_LABEL_SIZE, group=Groups.RENDERING)
    ] = None,
    show_bounds: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    show_axes: Annotated[bool | None, Parameter(group=Groups.RENDERING)] = None,
    zoom: Annotated[float | str | None, Parameter(group=Groups.RENDERING)] = None,
    border: Annotated[bool, Parameter(group=Groups.PLOTTER)] = False,
    border_color: Annotated[str, Parameter(group=Groups.PLOTTER)] = 'k',
    border_width: Annotated[float, Parameter(group=Groups.PLOTTER)] = 2.0,
    **kwargs: Annotated[
        Any,
        Parameter(help=HELP_KWARGS, converter=_kwargs_converter, group=Groups.SUPP),
    ],
) -> None:
    """Compare two or more mesh files side-by-side."""
    meshes = read_meshes(paths, skip_unreadable=skip_unreadable)
    if len(meshes) < 2:
        msg = (
            f'At least two readable paths are needed to compare, got {len(meshes)}.\n'
            f'Use `{pv.__name__} plot` to plot a single file.'
        )
        print_error_and_exit(message=msg)

    # Label each subplot with the name of the file it was read from
    names = labels if labels is not None else [path.stem for path in validate_paths(paths)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call_or_exit(
            pv.plot_compare,
            command='compare',
            datasets=meshes,
            labels=names,
            shape=None if shape is None else _parse_shape(shape),
            link=link,
            cpos=cpos,
            # The outline of a `MultiBlock` encloses every one of its blocks
            reference_mesh=pv.MultiBlock(meshes).outline() if outline else None,
            text_kwargs=None if label_size is None else {'font_size': label_size},
            show_bounds=show_bounds,
            show_axes=show_axes,
            zoom=zoom,
            screenshot=screenshot,
            plotter_kwargs={
                'off_screen': off_screen,
                'window_size': window_size,
                'border': border,
                'border_color': border_color,
                'border_width': border_width,
            },
            show_kwargs={'full_screen': full_screen, 'interactive': interactive},
            display_kwargs=kwargs,
        )

    _report_warnings(caught)
    return result
