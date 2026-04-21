"""`pyvista convert file.in file.out` CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
import warnings

from cyclopts import Parameter
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

import pyvista as pv

from .app import app
from .utils import HELP_FORMATTER
from .utils import _check_paths_exist
from .utils import _console_error


def _is_extension_only(file_out: str) -> bool:
    """Return ``True`` when ``file_out`` specifies only an extension (``.pv`` or ``dir/.pv``)."""
    path = Path(file_out)
    return not path.suffix and path.stem.startswith('.')


def _validate_out_has_extension(file_out: str) -> None:
    if not Path(file_out).suffix and not _is_extension_only(file_out):
        msg = 'Output file must have a file extension.'
        raise ValueError(msg)


@app.command(
    usage=f'Usage: [bold]{pv.__name__} convert FILE-IN [FILE-IN...] FILE-OUT',
    help_formatter=HELP_FORMATTER,
)
def _convert(
    files: Annotated[
        list[str],
        Parameter(
            name='files',
            consume_multiple=True,
            negative='',
            help=(
                'One or more input files followed by the output spec. '
                'Inputs may include glob patterns (e.g. ``*.vtu``) and must be readable '
                'with ``pyvista.read``. The final token is the output: a full filename '
                '(``bar.xyz``) when converting a single input, or an extension-only spec '
                '(``.xyz`` or ``dir/.xyz``) which reuses each input stem.'
            ),
        ),
    ],
) -> None:
    """Convert a mesh file to another format.

    One or more inputs may be supplied, and glob patterns are expanded. When multiple
    inputs are given, the output must be an extension-only spec (``.xyz`` or
    ``dir/.xyz``) so each input's stem is reused.

    Sample usage:
    ```bash
    pyvista convert foo.abc bar.xyz
    Saved: bar.xyz

    pyvista convert foo.abc .xyz
    Saved: foo.xyz

    pyvista convert *.vtu .pv
    Saved: a.pv
    Saved: b.pv
    ...

    pyvista convert *.vtu out/.pv
    Saved: out/a.pv
    Saved: out/b.pv
    ...
    ```
    """
    if len(files) < 2:
        _console_error(
            app=app,
            message='convert requires at least one input file and an output spec.',
        )

    file_out = files[-1]
    file_in_tokens = files[:-1]

    try:
        _validate_out_has_extension(file_out)
    except ValueError as e:
        _console_error(app=app, message=str(e))

    try:
        input_paths = _check_paths_exist(file_in_tokens)
    except ValueError as e:
        _console_error(app=app, message=str(e))

    path_out = Path(file_out)
    ext_only = _is_extension_only(file_out)
    if not ext_only and len(input_paths) > 1:
        _console_error(
            app=app,
            message=(
                f'Cannot write {len(input_paths)} inputs to a single named output '
                f'{file_out!r}. Use an extension-only output spec (e.g. '
                f"'{path_out.suffix}') to reuse each input's stem."
            ),
        )

    if len(input_paths) > 1:
        _convert_many(input_paths, path_out)
    else:
        _convert_one(input_paths[0], path_out, ext_only=ext_only, announce=True)


def _read_mesh(path_in: Path) -> pv.DataObject:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pv.InvalidMeshWarning)
        return pv.read(path_in)


def _convert_one(
    path_in: Path,
    path_out: Path,
    *,
    ext_only: bool,
    announce: bool,
) -> None:
    out_path = path_out.parent / f'{path_in.stem}{path_out.stem}' if ext_only else path_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = _read_mesh(path_in)
    except Exception as e:  # noqa: BLE001
        _console_error(app=app, message=f'Failed to read input file: {path_in}\n{e}')

    try:
        mesh.save(out_path)
    except Exception as e:  # noqa: BLE001
        _console_error(app=app, message=f'Failed to save output file: {out_path}\n{e}')

    if announce:
        app.console.print(f'[green]Saved:[/green] {out_path}')


def _convert_many(input_paths: list[Path], path_out: Path) -> None:
    columns = (
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('<'),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=app.console, transient=False) as progress:
        task = progress.add_task('Converting', total=len(input_paths))
        for path_in in input_paths:
            progress.update(task, description=f'Converting [cyan]{path_in.name}[/cyan]')
            _convert_one(path_in, path_out, ext_only=True, announce=False)
            progress.update(task, advance=1)

    app.console.print(
        f'[green]Saved {len(input_paths)} files to:[/green] {path_out.parent or Path()}/'
    )
