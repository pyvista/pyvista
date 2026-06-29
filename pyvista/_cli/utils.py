"""Utilities for command line interface.

Mostly contains converters, validators, console error helper and help formatters.

"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal
from typing import NamedTuple
from typing import NoReturn
from typing import get_args
import warnings

from cyclopts import Parameter
from cyclopts.help import ColumnSpec
from cyclopts.help import DefaultFormatter
from cyclopts.help import HelpEntry
from cyclopts.help import TableSpec
from rich import box
from rich.panel import Panel
from rich.text import Text

import pyvista as pv
from pyvista import _validation

from .app import CLI_APP

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console
    from rich.console import ConsoleOptions
    from rich.console import Group

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


class _MeshAndPath(NamedTuple):
    mesh: DataObject | None
    path: Path


def _console_error(message: str | Group, *, title: str = 'PyVista Error') -> NoReturn:
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
            matches = sorted(glob(v, recursive=True))  # noqa: PTH207
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(v)
        else:
            expanded.append(v)
    return expanded


def _validate_paths(paths: list[str]) -> list[Path]:
    """Expand globs and verify each path exists.

    Prints a console error and exists, listing any missing paths.
    """
    values = _expand_globs(paths)
    if not all((files := {v: Path(v).exists() for v in values}).values()):
        missing: str | list[str] = [k for k, v in files.items() if not v]
        n_missings = len(missing)

        literal_file = 'file' if n_missings == 1 else 'files'
        missing = missing[0] if n_missings == 1 else missing

        msg = f'{n_missings} {literal_file} not found: {missing}'
        _console_error(message=msg)
    return [Path(v) for v in values]


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


_ReadMeshOptions = Literal['exit', 'exit+hint', 'suppress', 'suppress+warn']


def _read_mesh(
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
            _console_error(message=msg)


class MeshPaths:
    def __init__(self, paths: list[str], *, skip_unreadable: bool, announce: bool) -> None:
        self._skip_unreadable = skip_unreadable
        self._announce = announce

        inital_paths = _validate_paths(paths)
        input_paths, dropped_paths = _filter_multiblock_children(inital_paths)
        self.paths: list[Path] = input_paths
        self._paths_dropped: list[Path] = dropped_paths
        self._print_dropped_multiblock_sidecar_dirs()

    def __iter__(self) -> Iterator[_MeshAndPath]:
        for path in self.paths:
            mesh = _read_mesh(
                path,
                on_error=('suppress+warn' if self._announce else 'suppress')
                if self._skip_unreadable
                else 'exit+hint',
            )
            yield _MeshAndPath(mesh=mesh, path=path)

    def _print_dropped_multiblock_sidecar_dirs(self) -> None:
        dropped = self._paths_dropped
        if n_dropped := len(dropped):
            listed = ', '.join(p.as_posix() for p in dropped[:5])
            if n_dropped > 5:
                listed += f', ... ({n_dropped - 5} more)'
            s = 's' if n_dropped > 1 else ''
            msg = (
                f'[yellow]Skipping {n_dropped} file{s} inside MultiBlock sidecar '
                f'directories:[/yellow] {listed}'
            )
            CLI_APP.console.print(msg)
