"""`pyvista compare file1.vtp file2.vtp --link=False ...` CLI."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
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
from .utils import LabelPosition
from .utils import LabelSize
from .utils import _kwargs_converter
from .utils import _validator_window_size
from .utils import call_or_exit
from .utils import print_error_and_exit
from .utils import read_meshes
from .utils import skip_unreadable
from .utils import validate_paths

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TextIO

_HELP_SHAPE = """\
Shape of the subplot grid, as either the number of rows and columns, e.g. ``2,2``,
or one of the string descriptors accepted by ``Plotter``, e.g. ``3|1`` for three
subplots on the left and one on the right. By default, a compact grid which is
never taller than it is wide is used.
"""

_HELP_LABELS = """\
Labels to show in each subplot. Must be given once per path. By default, the file name
of each path is used, with as much of the path as it takes to tell them apart.
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

_HELP_NORMALIZE = """\
Resize every mesh to a diagonal length of one, centered on the origin, so that meshes
of very different sizes are compared shape by shape. The files themselves are left as
they are. Normalized meshes are all the same size, so they share a camera by default.
An ``--outline`` says much less about them, since each is resized by a factor of its own.
"""

_HELP_LABEL_POSITION = """\
Where in each subplot to draw its label. Defaults to the upper left.
"""

_HELP_LABEL_SIZE = """\
Size of the label shown in each subplot, as either a font size or how to work one out.
A font size is used as given, and may be too large for a label to fit in its subplot.
``best_fit`` draws each label as large as it fits in its own subplot, and ``uniform``
draws them all at the size of the one which has to be smallest to fit. By default,
``uniform`` is used when the subplots are all the same size, and ``best_fit`` otherwise.
"""

# What to do about a dataset which is too small to make out, in terms of the options
# this command has rather than the arguments of `plot_compare`, which it does not
_REMEDIES = {
    'reference mesh': 'Omit `--outline` to fit each subplot to its own mesh.',
    'shared camera': (
        'Use `--no-link` to fit each subplot to its own mesh, or `--normalize` to '
        'resize them all to the same size.'
    ),
}


def _showwarning_with_advice(fallback: Callable[..., None]) -> Callable[..., None]:
    """Return a ``showwarning`` which prints a warning as it is raised, with advice.

    A warning is otherwise not seen until the interactive window this command opens is
    closed, since `plot_compare` raises every one of them well before it shows the
    window, but nothing is printed until the window closes and this call returns.
    """

    def showwarning(  # noqa: PLR0917
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        problem, made_out, _ = text.partition('make out.')
        remedy = next((advice for cause, advice in _REMEDIES.items() if cause in text), None)
        # This command draws the reference mesh itself, so name it as it is spelled here
        problem = problem.replace('the reference mesh', 'the outline')
        if not made_out or remedy is None:
            # Not a warning this command has anything better to say about. Fall back to
            # whichever `showwarning` was active before this one took over, rather than
            # raising the warning again, which would only call this same function once
            # more.
            fallback(message, category, filename, lineno, file, line)
            return
        CLI_APP.error_console.print(
            Panel(
                f'{problem}{made_out} {remedy}',
                style='magenta',
                title='Warning',
                title_align='left',
            )
        )

    return showwarning


def _label_paths(paths: list[Path]) -> list[str]:
    """Return the shortest name of each path which tells them all apart.

    Comparing ``file.vtk`` with ``file.vtp``, or ``run1/out.vtk`` with
    ``run2/out.vtk``, needs more than the stem of each path to say which subplot is
    which, but the extension and the directory are noise when they are all the same.
    """
    for name_of in (lambda path: path.stem, lambda path: path.name, str):
        labels = [name_of(path) for path in paths]
        if len(set(labels)) == len(labels):
            return labels
    # Nothing tells apart the same path given more than once
    return [str(path) for path in paths]


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
    sort_key=1,
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
    normalize: Annotated[bool, Parameter(help=_HELP_NORMALIZE, group=Groups.RENDERING)] = False,
    label_size: Annotated[
        float | LabelSize | None, Parameter(help=_HELP_LABEL_SIZE, group=Groups.RENDERING)
    ] = None,
    label_position: Annotated[
        LabelPosition | None, Parameter(help=_HELP_LABEL_POSITION, group=Groups.RENDERING)
    ] = None,
    show_bounds: Annotated[bool, Parameter(group=Groups.RENDERING)] = False,
    show_axes: Annotated[bool | None, Parameter(group=Groups.RENDERING)] = None,
    zoom: Annotated[float | str | None, Parameter(group=Groups.RENDERING)] = None,
    border: Annotated[bool, Parameter(group=Groups.PLOTTER)] = False,
    border_color: Annotated[str | None, Parameter(group=Groups.PLOTTER)] = None,
    border_width: Annotated[float | None, Parameter(group=Groups.PLOTTER)] = None,
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
    names = labels if labels is not None else _label_paths(validate_paths(paths))

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        # `catch_warnings` saves and restores `showwarning` along with the filter
        # above, so capture whichever one is active now to fall back to, print
        # warnings this command has nothing to add to as it would have anyway.
        warnings.showwarning = _showwarning_with_advice(warnings.showwarning)
        return call_or_exit(
            pv.plot_compare,
            command='compare',
            datasets=meshes,
            labels=names,
            shape=None if shape is None else _parse_shape(shape),
            link=link,
            cpos=cpos,
            # The outline of a `MultiBlock` encloses every one of its blocks
            reference_mesh=pv.MultiBlock(meshes).outline() if outline else None,
            normalize=normalize,
            label_size=label_size,
            label_position=label_position,
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
            dataset_kwargs=kwargs,
        )
