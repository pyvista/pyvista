"""Utilities for command line interface.

Mostly contains converters, validators and console error helper.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import NoReturn

from rich import box
from rich.panel import Panel

import pyvista
from pyvista.core.dataobject import DataObject

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cyclopts import App
    from cyclopts import Token
    from rich.console import Group

    from pyvista import DataObject


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
            mesh = pyvista.read(file)
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
