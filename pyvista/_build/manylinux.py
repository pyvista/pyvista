"""Run the build inside a manylinux container so the Linux wheel is portable."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import sys
from typing import TYPE_CHECKING

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

IMAGES = {
    'x86_64': 'quay.io/pypa/manylinux_2_28_x86_64',
    'aarch64': 'quay.io/pypa/manylinux_2_28_aarch64',
}
WORK = '/work'
SOURCE = '/src'
CACHE = '/cache'
PYVISTA = '/pyvista'
CCACHE_VERSION = '4.14'
CCACHE_SHA256 = '888eaf9697919d6eba77be797e9d9c9e40e734d11d67d25a79b075c3f79f6c77'
DOCKER_INSTRUCTIONS = 'Install Docker: https://docs.docker.com/get-docker/'


def host_arch() -> str:
    """Return the container architecture matching this machine.

    Returns
    -------
    str
        ``'x86_64'`` or ``'aarch64'``.

    """
    machine = platform.machine().lower()
    return 'aarch64' if machine in ('arm64', 'aarch64') else 'x86_64'


def default_image() -> str:
    """Return the manylinux image for this machine's architecture.

    Returns
    -------
    str
        Image reference.

    """
    return IMAGES[host_arch()]


def container_python() -> str:
    """Return the interpreter inside the image that matches the running Python.

    Returns
    -------
    str
        Path such as ``/opt/python/cp314-cp314/bin/python``.

    """
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    return f'/opt/python/{tag}-{tag}/bin/python'


def pyvista_source() -> Path | None:
    """Return the PyVista source tree the running package was installed from, if any.

    Returns
    -------
    Path or None
        Checkout root when PyVista runs from an editable install, else ``None``.

    """
    root = Path(pv.__file__).resolve().parent.parent
    if (root / 'pyproject.toml').is_file() and (root / 'pyvista' / '__init__.py').is_file():
        return root
    return None


def pyvista_requirement(source: Path | None) -> str:
    """Return what ``pip install`` gets inside the container to provide PyVista.

    Parameters
    ----------
    source : Path or None
        Host checkout mounted at :data:`PYVISTA`, or ``None`` to install a release.

    Returns
    -------
    str
        Requirement string or container path.

    """
    if source is not None:
        return PYVISTA
    version = pv.__version__
    return 'pyvista' if 'dev' in version else f'pyvista=={version}'


def container_script(
    *,
    python: str,
    packages: Sequence[str],
    pyvista: str,
    build_args: Sequence[str],
    ccache: bool,
    owner: str | None,
    paths: Sequence[str],
) -> str:
    """Return the shell script the container runs.

    Parameters
    ----------
    python : str
        Interpreter inside the image.

    packages : Sequence[str]
        System packages to install with ``dnf`` or ``yum``.

    pyvista : str
        Requirement or path that ``pip install`` gets PyVista from.

    build_args : Sequence[str]
        Arguments after ``pyvista build``.

    ccache : bool
        Whether to install a static ``ccache`` (x86_64 images only).

    owner : str or None
        ``uid:gid`` to hand the written files back to, or ``None`` to leave them.

    paths : Sequence[str]
        Mounted directories to hand back.

    Returns
    -------
    str
        Bash script.

    """
    lines = [
        'set -euo pipefail',
        f'PY={shlex.quote(python)}',
        'if ! test -x "$PY"; then echo "$PY is not in this image" >&2; exit 1; fi',
    ]
    if packages:
        pkgs = ' '.join(shlex.quote(p) for p in packages)
        lines.append(
            'if command -v dnf >/dev/null; then '
            f'dnf install -y --setopt=install_weak_deps=False {pkgs}; '
            f'else yum install -y {pkgs}; fi'
        )
    if ccache and host_arch() == 'x86_64':
        name = f'ccache-{CCACHE_VERSION}-linux-x86_64-musl-static'
        url = f'https://github.com/ccache/ccache/releases/download/v{CCACHE_VERSION}/{name}.tar.xz'
        lines += [
            f'curl -sSfL -o /tmp/ccache.tar.xz {url}',
            f'echo "{CCACHE_SHA256}  /tmp/ccache.tar.xz" | sha256sum --check --quiet',
            f'tar -xJf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 {name}/ccache',
        ]
    lines += [
        '"$PY" -m pip install --quiet --upgrade pip',
        f'"$PY" -m pip install --quiet --only-binary :all: {shlex.quote(pyvista)}',
        f'"$PY" -m pyvista build {shlex.join(build_args)}',
    ]
    if owner is not None and paths:
        lines.append(f'chown -R {owner} {" ".join(paths)}')
    return '\n'.join(lines)


def docker_command(
    image: str,
    *,
    mounts: Mapping[str, Path],
    script: str,
    interactive: bool,
    env: Mapping[str, str],
) -> list[str]:
    """Assemble the ``docker run`` command.

    Parameters
    ----------
    image : str
        Image to run.

    mounts : Mapping[str, Path]
        Container path to host directory.

    script : str
        Script passed to ``bash -c`` inside the container.

    interactive : bool
        Attach a terminal so the build inside can ask questions.

    env : Mapping[str, str]
        Host environment; ``CCACHE_*`` and ``SCCACHE_*`` variables are passed through.

    Returns
    -------
    list[str]
        Command and arguments.

    """
    cmd = ['docker', 'run', '--rm', '-i']
    if interactive:
        cmd.append('-t')
    for inside, outside in mounts.items():
        cmd += ['--volume', f'{outside}:{inside}']
    for name in sorted(env):
        if name.startswith(('CCACHE_', 'SCCACHE_')):
            cmd += ['--env', name]
    cmd += ['--workdir', WORK, image, 'bash', '-c', script]
    return cmd


def file_owner() -> str | None:
    """Return ``uid:gid`` of the current user, or ``None`` where that has no meaning.

    Returns
    -------
    str or None
        Owner spec for ``chown``.

    """
    getuid = getattr(os, 'getuid', None)
    getgid = getattr(os, 'getgid', None)
    if getuid is None or getgid is None:
        return None
    return f'{getuid()}:{getgid()}'
