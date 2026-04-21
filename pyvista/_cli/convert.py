"""`pyvista convert file.in file.out` CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

import pyvista as pv

from .app import app
from .utils import HELP_FORMATTER
from .utils import _console_error
from .utils import _load_paths


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
        inputs = _load_paths(file_in_tokens)
    except ValueError as e:
        _console_error(app=app, message=str(e))

    path_out = Path(file_out)
    ext_only = _is_extension_only(file_out)
    if not ext_only and len(inputs) > 1:
        _console_error(
            app=app,
            message=(
                f'Cannot write {len(inputs)} inputs to a single named output '
                f'{file_out!r}. Use an extension-only output spec (e.g. '
                f"'{path_out.suffix}') to reuse each input's stem."
            ),
        )

    for item in inputs:
        out_path = path_out.parent / f'{item.path.stem}{path_out.stem}' if ext_only else path_out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            item.mesh.save(out_path)
        except Exception as e:  # noqa: BLE001
            _console_error(app=app, message=f'Failed to save output file: {out_path}\n{e}')

        app.console.print(f'[green]Saved:[/green] {out_path}')
