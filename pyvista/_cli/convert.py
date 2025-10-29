"""`pyvista convert file.in file.out` CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

import pyvista

from .app import app
from .utils import HELP_FORMATTER
from .utils import _console_error
from .utils import _converter_files


def _validator_has_extension(type_: type, value: str) -> None:  # noqa: ARG001
    path = Path(value)
    has_suffix = bool(path.suffix)
    is_suffix = not path.suffix and path.stem.startswith('.')
    if not (has_suffix or is_suffix):
        msg = '\nOutput file must have a file extension.'
        raise ValueError(msg)


@app.command(
    usage=f'Usage: [bold]{pyvista.__name__} convert FILE-IN FILE-OUT',
    help_formatter=HELP_FORMATTER,
)
def _convert(
    file_in: Annotated[
        str,
        Parameter(
            help='File to convert. Must be readable with ``pyvista.read``.',
            converter=_converter_files,
        ),
    ],
    file_out: Annotated[
        str,
        Parameter(
            help='Output file. If only an file extension is given, '
            'the output has the same name as the input.',
            validator=_validator_has_extension,
        ),
    ],
) -> None:
    """Convert a mesh file to another format.

    Sample usage:
    ```bash
    pyvista convert foo.abc bar.xyz
    Saved: bar.xyz

    pyvista convert foo.abc .xyz
    Saved: foo.xyz
    ```
    """
    # get input mesh and input path from file_in str token
    # which was converted to a (mesh, path) pair
    mesh_in = file_in[0].mesh  # type: ignore[attr-defined]
    path_in = file_in[0].path  # type: ignore[attr-defined]

    # Parse output specification
    path_out = Path(file_out)
    out_dir = path_out.parent
    if not path_out.suffix:
        # Extension-only, use the input stem
        out_stem = path_in.stem
        out_suffix = path_out.stem
    else:
        # Explicit filename provided
        out_stem = path_out.stem
        out_suffix = path_out.suffix

    # Construct final output path
    out_path = out_dir / f'{out_stem}{out_suffix}'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh_in.save(out_path)
    except Exception as e:  # noqa: BLE001
        _console_error(app=app, message=f'Failed to save output file: {out_path}\n{e}')

    app.console.print(f'[green]Saved:[/green] {out_path}')
