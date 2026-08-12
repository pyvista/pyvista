"""Utilities for command line interface.

Mostly contains converters, validators, console error helper and help formatters.

"""

from __future__ import annotations

import ast
import glob
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import NoReturn
from typing import get_args
import warnings

from cyclopts import Parameter
from cyclopts import Token
from cyclopts.help import ColumnSpec
from cyclopts.help import DefaultFormatter
from cyclopts.help import HelpEntry
from cyclopts.help import TableSpec
from rich import box
from rich.console import Group
from rich.console import NewLine
from rich.panel import Panel
from rich.text import Text

import pyvista as pv
from pyvista import _validation
from pyvista.core.utilities.misc import StrEnum  # type: ignore [attr-defined]

from .app import CLI_APP

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from rich.console import Console
    from rich.console import ConsoleOptions

    from pyvista import DataObject


def default(entry: HelpEntry):  # noqa: ANN202
    return d if (d := entry.default) is not None else '-'


def names(entry: HelpEntry):  # noqa: ANN202
    strings = (*entry.names, *entry.shorts)
    names = Text(' '.join(strings), style='cyan')
    return (Text('* ', style='red') + names) if entry.required else names


def description(entry: HelpEntry):  # noqa: ANN202
    return entry.description


class _PyvistaHelpFormatter(DefaultFormatter):
    def render_usage(self, console: Console, options: ConsoleOptions, usage: str) -> None:  # noqa: ARG002
        """Render the usage line."""
        if usage:  # pragma: no branch
            console.print(usage)


HELP_FORMATTER = _PyvistaHelpFormatter(
    table_spec=TableSpec(show_header=True),
    column_specs=(
        ColumnSpec(
            renderer=names,
            header='Option',
            header_style='bold purple',
            style='cyan',
        ),
        ColumnSpec(
            renderer=default,
            header='Default',
            style='bold',
            header_style='bold purple',
        ),
        ColumnSpec(
            renderer=description,
            header='Description',
            header_style='bold purple',
        ),
    ),
)

_skip_unreadable_help = """
Skip any paths that are not readable instead of raising an error.
"""
skip_unreadable = Annotated[
    bool,
    Parameter(
        name='skip-unreadable',
        negative='',
        help=_skip_unreadable_help,
    ),
]


def print_error_and_exit(message: str | Group, *, title: str = 'PyVista Error') -> NoReturn:
    panel = Panel(
        message,
        title=title,
        style='bold red',
        box=box.ROUNDED,
        expand=True,
        title_align='left',
    )
    CLI_APP.error_console.print(panel)
    raise SystemExit(1)


_GLOB_CHARS = ('*', '?', '[')


def _expand_globs(values: list[str]) -> list[str]:
    """Expand any glob patterns in-place, preserving order.

    Tokens without glob characters are kept as-is so non-existent literals still raise the
    "file not found" error downstream. Glob patterns with no matches are kept so they surface
    as the missing token in the same error.
    """
    expanded: list[str] = []
    for v in values:
        v = str(Path(v).expanduser())  # noqa: PLW2901
        if any(c in v for c in _GLOB_CHARS):
            matches = sorted(glob.glob(v, recursive=True))  # noqa: PTH207
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(v)
        else:
            expanded.append(v)
    return expanded


def _check_paths_exist(paths: list[Path]) -> None:
    """Print a console error and exit if any paths do not exist."""
    files = {p: p.exists() for p in paths}
    if not all(files.values()):
        missing = [str(p) for p, exists in files.items() if not exists]
        n_missing = len(missing)

        literal_file = 'file' if n_missing == 1 else 'files'
        missing_display = missing[0] if n_missing == 1 else missing

        msg = f'{n_missing} {literal_file} not found: {missing_display}'
        print_error_and_exit(message=msg)


_MULTIBLOCK_EXTS = frozenset({'.vtm', '.vtmb'})


def _filter_multiblock_children(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Drop paths that live inside a sibling sidecar directory of a MultiBlock parent.

    A ``.vtm`` / ``.vtmb`` file ``parent.vtm`` is paired on disk with a sidecar directory
    ``parent/`` that holds the child blocks (e.g. ``parent/parent_0.vtp``). When both the
    parent and its sidecar children appear in the same input list, converting the children
    individually would duplicate work and break the 1:1 input/output mapping the user
    expects from a parent ``.vtm``.

    Children are filtered only when their parent is also in ``paths``. Standalone files
    inside an unrelated directory are left untouched.

    Returns a ``(kept, filtered)`` tuple; both lists preserve the input order.
    """
    sidecar_dirs = [
        (p.parent / p.stem).resolve() for p in paths if p.suffix.lower() in _MULTIBLOCK_EXTS
    ]
    if not sidecar_dirs:
        return paths, []

    kept: list[Path] = []
    filtered: list[Path] = []
    for p in paths:
        if p.suffix.lower() in _MULTIBLOCK_EXTS:
            kept.append(p)
            continue
        resolved_parents = set(p.resolve().parents)
        if any(sd in resolved_parents for sd in sidecar_dirs):
            filtered.append(p)
        else:
            kept.append(p)
    return kept, filtered


def validate_paths(paths: list[str]) -> list[Path]:
    """Expand globs, verify existence, and filter MultiBlock sidecar children.

    Prints a console message for any sidecar children that were filtered out.

    Returns
    -------
    list[Path]
        The validated input paths.

    """
    expanded = _expand_globs(paths)
    path_objects = [Path(v) for v in expanded]
    _check_paths_exist(path_objects)
    kept, dropped = _filter_multiblock_children(path_objects)
    if n_dropped := len(dropped):
        listed = ', '.join(p.as_posix() for p in dropped[:5])
        if n_dropped > 5:
            listed += f', ... ({n_dropped - 5} more)'
        s = 's' if n_dropped > 1 else ''
        msg = (
            f'[yellow]Skipping {n_dropped} file{s} inside MultiBlock sidecar '
            f'directories:[/yellow] {listed}'
        )
        CLI_APP.error_console.print(msg)
    return kept


_ReadMeshOptions = Literal['exit', 'exit+hint', 'suppress', 'suppress+warn']


def read_mesh(
    path: Path,
    *,
    on_error: _ReadMeshOptions = 'exit',
) -> DataObject | None:
    """Read a mesh with optional handling for read errors.

    Parameters
    ----------
    path
        Path to read with pyvista.read.

    on_error
        Behavior when the path cannot be read:

        - ``'exit'``: print a console error and call SystemExit (default).
        - ``'exit+hint'``: same as ``'exit'``, but append a hint to the error message
          indicating to use the ``--skip-unreadable`` option.
        - ``'suppress'``: return ``None`` silently.
        - ``'suppress+warn'``: return ``None`` and print a console error indicating the path is
          not readable.

    Returns
    -------
    DataObject | None
        Mesh object or None, depending on ``on_error`` value.

    """
    _validation.check_contains(get_args(_ReadMeshOptions), must_contain=on_error, name='on_error')
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pv.InvalidMeshWarning)
            return pv.read(path)
    except Exception:  # noqa: BLE001
        if on_error.startswith('suppress'):
            if on_error == 'suppress+warn':
                CLI_APP.error_console.print(f'[yellow]Skipping unreadable file:[/yellow] {path}')
            return None
        else:
            msg = f'Path is not readable by PyVista:\n{path}'
            if on_error == 'exit+hint':
                msg += '\nUse --skip-unreadable to skip this file.'
            print_error_and_exit(message=msg)


# The camera positions which can be named on the command line, as opposed to the fully
# specified positions which the Python API also accepts. Mirrors the keys of
# ``Renderer.CAMERA_STR_ATTR_MAP``
CposView = Literal['xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso']

# The ways a label size can be worked out, as opposed to the font size which may be
# named instead. Mirrors the modes of ``pyvista.plot_compare``.
LabelSize = Literal['best_fit', 'uniform']

# The places a label may be drawn in. Mirrors the places `Plotter.add_text` names,
# which is what ``pyvista.plot_compare`` draws its labels in.
LabelPosition = Literal[
    'lower_left',
    'lower_right',
    'upper_left',
    'upper_right',
    'lower_edge',
    'upper_edge',
    'left_edge',
    'right_edge',
]


class Groups(StrEnum):
    """Groups for the arguments of the plotting CLI commands."""

    PLOTTER = 'Plotter init'
    RENDERING = 'Rendering'
    SUPP = 'Supplementary'
    IN = 'Inputs'
    RETURN = 'Return'


def _validator_window_size(type_: type, value: list[int] | None) -> None:  # noqa: ARG001
    """Check that a window size is a pair of integers."""
    if value is not None and len(value) != 2:
        msg = 'Window size must be a list of two integers.'
        raise ValueError(msg)


def _kwargs_converter(type_, tokens: Sequence[Token]):  # noqa: ANN001, ANN202, ARG001
    """Coerce supplementary keyword arguments to Python values."""
    for token in tokens:
        # Check hyphen in keyword value
        if (h := '-') in (key := token.keys[0]):
            msg = f'A hyphen `{h}` has been used as supplementary keyword argument and is not converted to underscore `_`. Did you mean --{key.replace("-", "_")}={token.value} ?'  # noqa: E501
            CLI_APP.error_console.print(
                Panel(
                    msg,
                    style='magenta',
                    title='Warning',
                    title_align='left',
                )
            )

        # Coerce using literal_eval with fallback to str value
        try:
            return ast.literal_eval(token.value)
        except (ValueError, SyntaxError):
            return token.value
    return None


HELP_KWARGS = """\
Additional keyword arguments passed to ``Plotter.add_mesh`` or ``Plotter.add_volume``.
See the documentation for more details at https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh
and https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_volume

Note that contrary to other CLI arguments, hyphens ``-`` are not converted to underscores ``_``
before being passed to the corresponding plotter method. For example, you need to use
``--show_edges=True`` instead of ``--show-edges=True`` to show mesh edges in the plotting window.

"""


def read_meshes(paths: list[str], *, skip_unreadable: bool) -> list[DataObject]:
    """Validate and read every path, dropping the unreadable ones when asked to.

    Parameters
    ----------
    paths
        Paths to validate and read.

    skip_unreadable
        Skip any path which cannot be read instead of exiting.

    Returns
    -------
    list[DataObject]
        Mesh read from each path.

    """
    valid_paths = validate_paths(paths)
    # Inform users about --skip-unreadable option when there are multiple inputs
    on_error: _ReadMeshOptions = 'exit+hint' if len(valid_paths) > 1 else 'exit'
    meshes = [
        read_mesh(path, on_error='suppress' if skip_unreadable else on_error)
        for path in valid_paths
    ]
    return [mesh for mesh in meshes if mesh is not None]


def call_or_exit(func: Callable[..., Any], /, command: str, **kwargs) -> Any:
    """Call a plotting function, reporting any exception it raises as a console error.

    Parameters
    ----------
    func
        Function to call.

    command
        Name of the CLI command, used to print its help along with the error.

    **kwargs
        Keyword arguments to call ``func`` with.

    Returns
    -------
    Any
        Whatever ``func`` returns.

    """
    try:
        return func(**kwargs)
    except Exception as ex:  # noqa: BLE001
        # Prevent traceback and output error along with help message
        CLI_APP.help_print(tokens=command, console=CLI_APP.error_console)

        called = f'pv.{func.__name__}'
        msg = Group(
            f':warning: The following exception has been raised when calling [u]{called}[/u]:',
            NewLine(),
            Panel(
                str(ex), title=f'{type(ex).__name__}', title_align='left', style='bold blink red'
            ),
            NewLine(),
            Text('Please check the provided arguments.'),
        )
        print_error_and_exit(message=msg)
