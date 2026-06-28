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
from .utils import skip_unreadable


def _is_extension_only(file_out: str) -> bool:
    """Return ``True`` when ``file_out`` specifies only an extension (``.pv`` or ``dir/.pv``)."""
    path = Path(file_out)
    return not path.suffix and path.stem.startswith('.')


def _validate_out_has_extension(file_out: str) -> None:
    """Raise ``ValueError`` when ``file_out`` is missing a file extension."""
    if not Path(file_out).suffix and not _is_extension_only(file_out):
        msg = 'Output file must have a file extension.'
        raise ValueError(msg)


_resolve_collisions_help = (
    'When multiple inputs resolve to the same output path, automatically rename '
    'colliding outputs by appending ``_1``, ``_2``, etc. to the stem. '
    'Without this flag, collisions are an error.'
)


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
    *,
    resolve_collisions: Annotated[
        bool,
        Parameter(
            name='--resolve-collisions',
            negative='',
            help=_resolve_collisions_help,
        ),
    ] = False,
    skip_unreadable: skip_unreadable = False,
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

    input_paths = MeshPaths(file_in_tokens, app=app, skip_unreadable=False, announce=False).paths

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
        _convert_many(
            input_paths,
            path_out,
            skip_unreadable=skip_unreadable,
            resolve_collisions=resolve_collisions,
        )
    else:
        _convert_one(
            input_paths[0],
            path_out,
            ext_only=ext_only,
            announce=True,
            skip_unreadable=skip_unreadable,
        )


def _resolve_out_path(path_in: Path, path_out: Path, *, ext_only: bool) -> Path:
    """Compute the per-input output path.

    A bare extension-only spec (``.pv``) is written next to each input. An extension-only
    spec with an explicit parent (``out/.pv``) is written into that parent.
    """
    if not ext_only:
        return path_out
    out_dir = path_in.parent if str(path_out.parent) == '.' else path_out.parent
    return out_dir / f'{path_in.stem}{path_out.stem}'


def _build_output_map(
    input_paths: list[Path],
    path_out: Path,
    *,
    resolve_collisions: bool,
) -> tuple[dict[Path, Path], list[tuple[Path, Path, Path]]]:
    """Return a mapping of input path → resolved output path, and a list of renames.

    When ``resolve_collisions`` is ``False`` and collisions exist, prints a console
    error and exits. When ``resolve_collisions`` is ``True``, colliding outputs are
    renamed by appending ``_1``, ``_2``, etc. to the stem.

    Returns a tuple of:
    - ``output_map``: input path → final output path
    - ``renames``: list of ``(path_in, original_out, renamed_out)`` for each rename
    """
    seen: dict[Path, Path] = {}  # output path → first input that claimed it
    counters: dict[Path, int] = {}  # base output path → next suffix counter
    result: dict[Path, Path] = {}  # input path → final output path
    renames: list[tuple[Path, Path, Path]] = []
    collisions: list[tuple[Path, Path, Path]] = []

    for path_in in input_paths:
        base_out = _resolve_out_path(path_in, path_out, ext_only=True)

        if base_out not in seen:
            seen[base_out] = path_in
            result[path_in] = base_out
        elif resolve_collisions:
            n = counters.get(base_out, 1)
            candidate = base_out.with_stem(f'{base_out.stem}_{n}')
            while candidate in seen:
                n += 1
                candidate = base_out.with_stem(f'{base_out.stem}_{n}')
            counters[base_out] = n + 1
            seen[candidate] = path_in
            result[path_in] = candidate
            renames.append((path_in, base_out, candidate))
        else:
            collisions.append((seen[base_out], path_in, base_out))

    if collisions:
        lines = '\n'.join(f'  {a.name} + {b.name} → {out}' for a, b, out in collisions)
        _console_error(
            app=app,
            message=(
                f'{len(collisions)} output collision(s) detected:\n{lines}\n'
                'Use --resolve-collisions to rename automatically.'
            ),
        )

    return result, renames


def _save_mesh(mesh: pv.DataObject, out_path: Path) -> None:
    """Save ``mesh`` to ``out_path``, creating parent directories as needed.

    Raises a console error on failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.save(out_path)
    except Exception as e:  # noqa: BLE001
        _console_error(app=app, message=f'Failed to save output file: {out_path}\n{e}')


def _convert_one(
    path_in: Path,
    path_out: Path,
    *,
    ext_only: bool,
    announce: bool,
    skip_unreadable: bool,
) -> bool:
    """Read a single input, save it under the resolved output path, and optionally announce."""
    out_path = _resolve_out_path(path_in, path_out, ext_only=ext_only)
    mesh = _read_mesh(
        path_in,
        app=app,
        skip_unreadable=skip_unreadable,
        announce_unreadable=announce,
        append_skip_unreadable_msg=False,
    )
    if mesh is None:
        return False
    _save_mesh(mesh, out_path)
    if announce:
        app.console.print(f'[green]Saved:[/green] {out_path}')
    return True


def _convert_many(
    input_paths: list[Path],
    path_out: Path,
    *,
    skip_unreadable: bool,
    resolve_collisions: bool,
) -> None:
    """Convert each input under a progress bar and report the destination directory(s)."""
    output_map, renames = _build_output_map(
        input_paths, path_out, resolve_collisions=resolve_collisions
    )

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
    n_converted = 0
    skipped: list[Path] = []

    with Progress(*columns, console=app.console, transient=False) as progress:
        task = progress.add_task('Converting', total=len(input_paths))
        for path_in, out_path in output_map.items():
            progress.update(task, description=f'Converting [cyan]{path_in.name}[/cyan]')
            mesh = _read_mesh(
                path_in,
                app=app,
                skip_unreadable=skip_unreadable,
                announce_unreadable=False,
                append_skip_unreadable_msg=True,
            )
            if mesh is None:
                skipped.append(path_in)
            else:
                _save_mesh(mesh, out_path)
                out_dirs.add(out_path.parent)
                n_converted += 1
            progress.update(task, advance=1)

    if len(out_dirs) == 1:
        app.console.print(f'[green]Saved {n_converted} files to:[/green] {next(iter(out_dirs))}/')
    else:
        app.console.print(
            f'[green]Saved {n_converted} files across {len(out_dirs)} directories.[/green]'
        )

    if renames:
        app.console.print(f'\n[yellow]{len(renames)} collision(s) resolved by renaming:[/yellow]')
        for path_in, original, renamed in renames:
            app.console.print(
                f'  {path_in.name} → {renamed.name} (would have collided with {original.name})'
            )

    if skipped:
        app.console.print(f'\n[yellow]{len(skipped)} file(s) skipped (unreadable):[/yellow]')
        for path_in in skipped:
            app.console.print(f'  {path_in}')
