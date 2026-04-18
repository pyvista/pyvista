"""Utilities for command line interface.

Mostly contains converters, validators, console error helper and help formatters.

"""

from __future__ import annotations

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
    from collections.abc import Sequence

    from cyclopts import App
    from cyclopts import Token
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


def _converter_files(
    type_: type,  # noqa: ARG001
    tokens: Sequence[Token],
) -> list[_MeshAndPath]:
    """Helper function used to read provided files.

    Raises errors if:

    - any file does not exits
    - any file is not readable with ``pv.read``

    """  # noqa: D401
    values: list[str] = [t.value for t in tokens]

    # Test file exists
    if not all((files := {v: Path(v).exists() for v in values}).values()):
        missing: str | list[str] = [k for k, v in files.items() if not v]
        n_missings = len(missing)

        literal_file = 'file' if n_missings == 1 else 'files'
        missing = missing[0] if n_missings == 1 else missing

        msg = f'{n_missings} {literal_file} not found: {missing}'
        raise ValueError(msg)

    # Test file can be read by pyvista
    meshes_and_paths: list[_MeshAndPath] = []
    not_readable: list[str] = []
    for file in values:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    category=pv.InvalidMeshWarning,
                )
                mesh = pv.read(file)
        except Exception:  # noqa: BLE001
            not_readable.append(file)
        else:
            meshes_and_paths.append(_MeshAndPath(mesh=mesh, path=Path(file)))

    if len(not_readable) > 0:
        n = len(not_readable)
        literal_file = 'file' if n == 1 else 'files'
        msg = f'{n} {literal_file} not readable by PyVista:\n{not_readable}'
        raise ValueError(msg)

    return meshes_and_paths
