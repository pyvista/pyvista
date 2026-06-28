"""`pyvista convert file.in file.out` CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
from .utils import MeshPaths
from .utils import _console_error
from .utils import _read_mesh


def _is_extension_only(file_out: str) -> bool:
    """Return ``True`` when ``file_out`` specifies only an extension (``.pv`` or ``dir/.pv``)."""
    path = Path(file_out)
    return not path.suffix and path.stem.startswith('.')


def _validate_out_has_extension(file_out: str) -> None:
    """Raise ``ValueError`` when ``file_out`` is missing a file extension."""
    if not Path(file_out).suffix and not _is_extension_only(file_out):
        msg = 'Output file must have a file extension.'
        raise ValueError(msg)


@app.command(
    usage=f'Usage: [bold]{pv.__name__} convert PATH-IN [PATH-IN...] PATH-OUT',
    help_formatter=HELP_FORMATTER,
)
def _convert(
    paths: Annotated[
        list[str],
        Parameter(
            consume_multiple=True,
            help=(
                'One or more input paths followed by the output spec. '
                'Inputs may include glob patterns (e.g. ``*.vtu``) and must be readable '
                'with ``pyvista.read``. The final token is the output: a full filename '
                '(``bar.xyz``) when converting a single input, or an extension-only spec '
                '(``.xyz`` or ``dir/.xyz``) which reuses each input stem. A bare ``.xyz`` '
                'writes adjacent to each input; ``dir/.xyz`` writes into ``dir/``.'
            ),
        ),
    ],
    /,
) -> None:
    """Convert a mesh file to another format.

    One or more inputs may be supplied, and glob patterns are expanded. When multiple
    inputs are given, the output must be an extension-only spec (``.xyz`` or
    ``dir/.xyz``) so each input's stem is reused. A bare ``.xyz`` writes the output
    adjacent to each input; ``dir/.xyz`` writes into the given ``dir``.

    MultiBlock files (``.vtm``/``.vtmb``) are paired on disk with a sibling sidecar
    directory (e.g. ``parent.vtm`` -> ``parent/parent_0.vtp``). Pass the parent
    ``.vtm`` directly to convert a MultiBlock as a single output; a recursive glob
    like ``**/*`` may also pick up the sidecar children. When both a parent and its
    sidecar children appear in the input list the children are dropped automatically
    to preserve the 1:1 input/output mapping.

    Sample usage:
    ```bash
    pyvista convert foo.abc bar.xyz
    Saved: bar.xyz

    pyvista convert foo.abc .xyz
    Saved: foo.xyz

    pyvista convert sub/*.vtu .pv
    # Writes sub/a.pv, sub/b.pv, ... next to each input

    pyvista convert sub/*.vtu out/.pv
    # Writes out/a.pv, out/b.pv, ... into the explicit out directory
    ```
    """
    if len(paths) < 2:
        _console_error(
            app=app,
            message='convert requires at least one input file and an output spec.',
        )

    file_out = paths[-1]
    file_in_tokens = paths[:-1]

    try:
        _validate_out_has_extension(file_out)
    except ValueError as e:
        _console_error(app=app, message=str(e))

    # Use MeshPath obj to validate input paths and handle mesh read errors
    input_paths = MeshPaths(file_in_tokens, app=app).paths

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


def _resolve_out_path(path_in: Path, path_out: Path, *, ext_only: bool) -> Path:
    """Compute the per-input output path.

    A bare extension-only spec (``.pv``) is written next to each input. An extension-only
    spec with an explicit parent (``out/.pv``) is written into that parent.
    """
    if not ext_only:
        return path_out
    out_dir = path_in.parent if str(path_out.parent) == '.' else path_out.parent
    return out_dir / f'{path_in.stem}{path_out.stem}'


def _convert_one(
    path_in: Path,
    path_out: Path,
    *,
    ext_only: bool,
    announce: bool,
) -> None:
    """Read a single input, save it under the resolved output path, and optionally announce."""
    out_path = _resolve_out_path(path_in, path_out, ext_only=ext_only)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = _read_mesh(path_in, app=app)

    try:
        mesh.save(out_path)
    except Exception as e:  # noqa: BLE001
        _console_error(app=app, message=f'Failed to save output file: {out_path}\n{e}')

    if announce:
        app.console.print(f'[green]Saved:[/green] {out_path}')


def _convert_many(input_paths: list[Path], path_out: Path) -> None:
    """Convert each input under a progress bar and report the destination directory(s)."""
    columns = (
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('<'),
        TimeRemainingColumn(),
    )
    out_dirs: set[Path] = set()
    with Progress(*columns, console=app.console, transient=False) as progress:
        task = progress.add_task('Converting', total=len(input_paths))
        for path_in in input_paths:
            progress.update(task, description=f'Converting [cyan]{path_in.name}[/cyan]')
            out_path = _resolve_out_path(path_in, path_out, ext_only=True)
            out_dirs.add(out_path.parent)
            _convert_one(path_in, path_out, ext_only=True, announce=False)
            progress.update(task, advance=1)

    if len(out_dirs) == 1:
        app.console.print(
            f'[green]Saved {len(input_paths)} files to:[/green] {next(iter(out_dirs))}/'
        )
    else:
        app.console.print(
            f'[green]Saved {len(input_paths)} files across {len(out_dirs)} directories.[/green]'
        )
