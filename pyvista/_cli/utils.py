"""Utilities for command line interface.

Mostly contains converters, validators, console error helper and help formatters.

"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn
import warnings

from cyclopts.help import ColumnSpec
from cyclopts.help import DefaultFormatter
from cyclopts.help import HelpEntry
from cyclopts.help import TableSpec
from rich import box
from rich.panel import Panel
from rich.text import Text

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cyclopts import App
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


class _MeshAndPath(NamedTuple):
    mesh: DataObject
    path: Path


def _console_error(*, app: App, message: str | Group, title: str = 'PyVista Error') -> NoReturn:
    panel = Panel(
        message,
        title=title,
        style='bold red',
        box=box.ROUNDED,
        expand=True,
        title_align='left',
    )
    app.error_console.print(panel)
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
        if any(c in v for c in _GLOB_CHARS):
            matches = sorted(glob(v, recursive=True))  # noqa: PTH207
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(v)
        else:
            expanded.append(v)
    return expanded


def _check_paths_exist(paths: list[str], app: App) -> list[Path]:
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
        _console_error(app=app, message=msg)
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


def _read_mesh(path: Path, app: App) -> DataObject:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=pv.InvalidMeshWarning,
            )
            return pv.read(path)
    except Exception:  # noqa: BLE001
        msg = f'Path is not readable by PyVista:\n{path}'
        _console_error(app=app, message=msg)


class MeshPaths:
    def __init__(self, paths: list[str], *, app: App) -> None:
        inital_paths = _check_paths_exist(paths, app=app)
        input_paths, dropped_paths = _filter_multiblock_children(inital_paths)
        self.paths: list[Path] = input_paths
        self._paths_dropped: list[Path] = dropped_paths

        self._app = app
        self._print_dropped_multiblock_sidecar_dirs()

    def __iter__(self) -> Iterator[_MeshAndPath]:
        for path in self.paths:
            mesh = _read_mesh(path, app=self._app)
            yield _MeshAndPath(mesh=mesh, path=path)

    def _print_dropped_multiblock_sidecar_dirs(self) -> None:
        dropped = self._paths_dropped
        if dropped:
            n = len(dropped)
            listed = ', '.join(str(p) for p in dropped[:5])
            if n > 5:
                listed += f', ... ({n - 5} more)'
            self._app.console.print(
                f'[yellow]Skipping {n} file(s) inside MultiBlock sidecar directories:[/yellow] '
                f'{listed}'
            )
