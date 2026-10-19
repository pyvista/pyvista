"""``pyvista build BACKEND`` CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts import validators
from rich.markup import escape

import pyvista as pv
from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import ManifestError
from pyvista._build.manylinux import default_image
from pyvista._build.phases import BUILD_TYPES
from pyvista._build.phases import COMPILER_CACHES
from pyvista._build.phases import BuildType
from pyvista._build.phases import CompilerCache
from pyvista._build.registry import known_backends
from pyvista._vtk import BACKEND_ENV_VAR

from .app import CLI_APP
from .utils import HELP_FORMATTER
from .utils import print_error_and_exit


def _split_csv(values: list[str] | None) -> list[str]:
    """Flatten repeated and comma-separated values into one list."""
    return [item.strip() for value in values or [] for item in value.split(',') if item.strip()]


def _choices(values: tuple[str, ...]) -> str:
    """Join choices into an English list with an Oxford comma."""
    return f'{", ".join(values[:-1])}, or {values[-1]}'


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
        str | None,
        Parameter(help="Git URL to clone instead of the backend's own, such as a fork."),
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
    minimal: Annotated[
        bool, Parameter(negative='', help='Enable only the required groups.')
    ] = False,
    opt: Annotated[
        list[str] | None,
        Parameter(negative='', help='Set a manifest option, ``NAME=VALUE``. Repeatable.'),
    ] = None,
    build_type: Annotated[
        BuildType,
        Parameter(help=f'CMake build type: {_choices(BUILD_TYPES)}.'),
    ] = 'Release',
    jobs: Annotated[
        int | None,
        Parameter(
            validator=validators.Number(gte=1),
            help='Parallel compile jobs. Defaults to one per CPU core, at most one per 2 GB of '
            'memory.',
        ),
    ] = None,
    define: Annotated[
        list[str] | None,
        Parameter(
            name=['--define', '-D'],
            negative='',
            help='Extra CMake cache entry, ``KEY=VALUE``, applied last so it overrides the '
            'manifest.',
        ),
    ] = None,
    clean: Annotated[bool, Parameter(negative='', help='Delete the build tree first.')] = False,
    reconfigure: Annotated[
        bool, Parameter(negative='', help='Run cmake again even if the tree is configured.')
    ] = False,
    update: Annotated[
        bool,
        Parameter(
            negative='',
            help='Fetch, then check out ``--ref`` (or the release tag) in an existing checkout.',
        ),
    ] = False,
    compiler_cache: Annotated[
        CompilerCache,
        Parameter(
            help=f'Compiler cache: {_choices(COMPILER_CACHES)}. ``auto`` uses whichever is on '
            'PATH.'
        ),
    ] = 'none',
    cache_dir: Annotated[
        Path | None,
        Parameter(
            help='Compiler cache directory, used with ``--compiler-cache``. Defaults to '
            "``$CCACHE_DIR`` or ``$SCCACHE_DIR`` when set, else the cache tool's own default "
            'location, or ``<build-dir>/compiler-cache`` with ``--manylinux``.'
        ),
    ] = None,
    build_dir: Annotated[
        Path | None,
        Parameter(
            help='Where source, build tree, and wheels go. Defaults to ``./<backend>-build``.'
        ),
    ] = None,
    version_suffix: Annotated[
        str | None,
        Parameter(
            help='Wheel version suffix. Defaults to ``dev0+g<sha>``, or none when the '
            'checkout is at a tag.'
        ),
    ] = None,
    repair: Annotated[
        bool,
        Parameter(
            help='Bundle the libraries the wheel links against and tag it for other machines. '
            '``--no-repair`` skips it.'
        ),
    ] = True,
    install: Annotated[
        bool, Parameter(negative='', help='Install the built wheel into this environment.')
    ] = False,
    yes: Annotated[
        bool,
        Parameter(
            name=['--yes', '-y'],
            negative='',
            help='Start, and install what the build needs, without asking.',
        ),
    ] = False,
    manylinux: Annotated[
        bool,
        Parameter(
            negative='',
            help='Build inside a manylinux container through Docker, for a portable Linux wheel.',
        ),
    ] = False,
    manylinux_image: Annotated[
        str | None,
        Parameter(help='Container image for ``--manylinux``. Defaults to manylinux_2_28.'),
    ] = None,
    interactive: Annotated[
        bool,
        Parameter(
            name=['--interactive', '-i'],
            negative='',
            help='Walk through the steps, confirming each one before any command runs.',
        ),
    ] = False,
    dry_run: Annotated[
        bool, Parameter(negative='', help='Print the commands and run nothing.')
    ] = False,
    verbose: Annotated[
        bool,
        Parameter(
            name=['--verbose', '-v'],
            negative='',
            help=(
                'Print each command with its environment variables, and all of its output '
                'instead of one updating line.'
            ),
        ),
    ] = False,
) -> None:
    """Build a VTK backend wheel from source.

    Asks once before starting, then builds with the defaults and prints every command as it
    runs, with its output on one updating line (``--verbose`` prints all of it, and the
    environment variables each command sets). Anything the
    build has to install is listed before that question, which covers it. ``--yes`` answers
    it.
    ``--interactive`` walks through the steps, showing each setting and asking before running
    anything, and ``--dry-run`` prints the commands without running them. ``--manylinux``
    runs the build inside a manylinux container so the Linux wheel installs on any machine.
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
    if manylinux_image is not None and not manylinux:
        print_error_and_exit('--manylinux-image needs --manylinux.')
    if cache_dir is not None and compiler_cache == 'none':
        print_error_and_exit('--cache-dir needs --compiler-cache.')
    config = BuildConfig(
        backend=backend,
        build_dir=(build_dir or Path(f'{backend}-build')).resolve(),
        source=source.resolve() if source is not None else None,
        url=url,
        ref=ref,
        with_=_split_csv(with_),
        without=_split_csv(without),
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
        interactive=interactive,
        dry_run=dry_run,
        verbose=verbose,
        manylinux=(manylinux_image or default_image()) if manylinux else None,
    )
    try:
        Builder(config, console=CLI_APP.console).build()
    except (BuildError, ManifestError, OSError) as err:
        print_error_and_exit(escape(str(err)), title='PyVista build stopped')
    except (KeyboardInterrupt, EOFError):
        CLI_APP.console.print('\n[dim]Stopped.[/dim]')
        raise SystemExit(130) from None
