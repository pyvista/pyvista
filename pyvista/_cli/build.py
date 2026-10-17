"""``pyvista build BACKEND`` CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated
from typing import Literal

from cyclopts import Parameter

import pyvista as pv
from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import ManifestError
from pyvista._build.registry import known_backends
from pyvista._vtk import BACKEND_ENV_VAR

from .app import CLI_APP
from .utils import HELP_FORMATTER
from .utils import print_error_and_exit


def _split_csv(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated values into one list."""
    return [item.strip() for value in values or [] for item in value.split(',') if item.strip()]


_backend_help = (
    f'Backend to build: one of {", ".join(known_backends())}, or any name together with '
    f'``--url`` or ``--source``. Defaults to ``${BACKEND_ENV_VAR}`` when that is set.'
)


@CLI_APP.command(
    sort_key=5,
    usage=f'Usage: [bold]{pv.__name__} build BACKEND [OPTIONS]',
    help_formatter=HELP_FORMATTER,
)
def _build(
    backend: Annotated[str | None, Parameter(help=_backend_help)] = None,
    *,
    ref: Annotated[
        str | None,
        Parameter(
            help="Git tag, branch or commit to build. Defaults to the manifest's release tag."
        ),
    ] = None,
    source: Annotated[
        Path | None,
        Parameter(help='Existing source checkout to build instead of cloning.'),
    ] = None,
    url: Annotated[
        str | None, Parameter(help='Git URL to clone, for a backend PyVista does not know.')
    ] = None,
    with_: Annotated[
        list[str] | None,
        Parameter(
            name='--with', negative='', help='Groups to enable, comma-separated or repeated.'
        ),
    ] = None,
    without: Annotated[
        list[str] | None,
        Parameter(
            name='--without', negative='', help='Groups to disable, comma-separated or repeated.'
        ),
    ] = None,
    all_: Annotated[
        bool, Parameter(name='--all', negative='', help='Enable every group available here.')
    ] = False,
    minimal: Annotated[
        bool, Parameter(negative='', help='Enable only the required groups.')
    ] = False,
    opt: Annotated[
        list[str] | None,
        Parameter(negative='', help='Set a manifest option, ``NAME=VALUE``; repeatable.'),
    ] = None,
    build_type: Annotated[
        Literal['Release', 'RelWithDebInfo', 'Debug'], Parameter(help='CMake build type.')
    ] = 'Release',
    jobs: Annotated[
        int | None, Parameter(help='Parallel compile jobs. Defaults to the CPU count.')
    ] = None,
    define: Annotated[
        list[str] | None,
        Parameter(
            name=['--define', '-D'],
            negative='',
            help='Extra CMake cache entry, ``KEY=VALUE``; applied last, overriding the manifest.',
        ),
    ] = None,
    clean: Annotated[bool, Parameter(negative='', help='Delete the build tree first.')] = False,
    reconfigure: Annotated[
        bool, Parameter(negative='', help='Run cmake again even if the tree is configured.')
    ] = False,
    update: Annotated[
        bool, Parameter(negative='', help='Fetch and check out ``--ref`` in an existing checkout.')
    ] = False,
    compiler_cache: Annotated[
        Literal['auto', 'ccache', 'sccache', 'none'],
        Parameter(help='Compiler launcher. ``auto`` uses whichever is on PATH.'),
    ] = 'auto',
    cache_dir: Annotated[
        Path | None,
        Parameter(help='Compiler cache directory. Defaults to ``<build-dir>/compiler-cache``.'),
    ] = None,
    build_dir: Annotated[
        Path | None,
        Parameter(
            help='Where source, build tree and wheels go. Defaults to ``./<backend>-build``.'
        ),
    ] = None,
    version_suffix: Annotated[
        str | None,
        Parameter(
            help='Wheel version suffix. Defaults to ``dev0+g<sha>`` unless ``--ref`` is a tag.'
        ),
    ] = None,
    repair: Annotated[
        bool, Parameter(help='Bundle shared libraries into the wheel (``--no-repair`` to skip).')
    ] = True,
    install: Annotated[
        bool, Parameter(help='Install the wheel into this environment without asking.')
    ] = False,
    yes: Annotated[
        bool, Parameter(negative='', help='Accept every default without asking.')
    ] = False,
    dry_run: Annotated[
        bool, Parameter(negative='', help='Print the commands and run nothing.')
    ] = False,
) -> None:
    """Build a VTK backend wheel from source.

    Interactive when run from a terminal: each step says what it will do and waits for a
    yes. Pass ``--yes`` to accept every default, or ``--dry-run`` to only print the commands.
    """
    if backend is None:
        backend = os.environ.get(BACKEND_ENV_VAR)
        if not backend:
            msg = (
                f'Say which backend to build, e.g. `pyvista build vtk`. '
                f'Known backends: {", ".join(known_backends())}.'
            )
            print_error_and_exit(msg)
    if backend == 'vtkmodules':
        backend = 'vtk'
    if source is not None and url is not None:
        print_error_and_exit('--source and --url are mutually exclusive.')
    config = BuildConfig(
        backend=backend,
        build_dir=(build_dir or Path(f'{backend}-build')).resolve(),
        source=source.resolve() if source is not None else None,
        url=url,
        ref=ref,
        with_=_split_csv(with_),
        without=_split_csv(without),
        all_=all_,
        minimal=minimal,
        build_type=build_type,
        jobs=jobs,
        defines=define or [],
        options=opt or [],
        clean=clean,
        reconfigure=reconfigure,
        update=update,
        compiler_cache=compiler_cache,
        cache_dir=cache_dir.resolve() if cache_dir is not None else None,
        version_suffix=version_suffix,
        repair=repair,
        install=install,
        yes=yes,
        dry_run=dry_run,
    )
    try:
        Builder(config, console=CLI_APP.console).build()
    except (BuildError, ManifestError) as err:
        print_error_and_exit(str(err), title='PyVista build stopped')
    except KeyboardInterrupt:
        print_error_and_exit('Interrupted.', title='PyVista build stopped')
