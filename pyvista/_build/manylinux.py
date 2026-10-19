"""Run the build inside a manylinux container so the Linux wheel is portable."""

from __future__ import annotations

from importlib.metadata import requires
import os
from pathlib import Path
import platform
import shlex
import sys
from typing import TYPE_CHECKING

from packaging.requirements import Requirement

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
PYVISTA_ROOT = '/opt/pyvista-host'
PIP_CACHE = f'{WORK}/pip-cache'
CCACHE_VERSION = '4.14'
CCACHE_SHA256 = {
    'x86_64': '888eaf9697919d6eba77be797e9d9c9e40e734d11d67d25a79b075c3f79f6c77',
    'aarch64': '40623f11012a97d60c7d90058d7e3c8c70a3802af53a34067b89e6345c89db69',
}
DOCKER_INSTRUCTIONS = 'Install Docker: https://docs.docker.com/get-docker/'
_HOST_PATH_VARS = frozenset(
    {'CCACHE_BASEDIR', 'CCACHE_CONFIGPATH', 'CCACHE_DIR', 'CCACHE_LOGFILE', 'SCCACHE_DIR'}
)


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
        Path such as ``/opt/python/cp314-cp314/bin/python``, or ``cp314-cp314t`` on a
        free-threaded build.

    """
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    abi = tag + ('t' if 't' in getattr(sys, 'abiflags', '') else '')
    return f'/opt/python/{tag}-{abi}/bin/python'


def pyvista_package() -> Path:
    """Return the directory of the running ``pyvista`` package.

    Returns
    -------
    Path
        Package directory, mounted read-only into the container.

    """
    return Path(__file__).resolve().parents[1]


def pyvista_requirements() -> list[str]:
    """Return PyVista's runtime requirements, without the optional extras.

    Returns
    -------
    list[str]
        Requirement strings for ``pip install`` inside the container.

    """
    names = requires('pyvista') or []
    return [r for r in names if 'extra' not in str(Requirement(r).marker or '')]


def container_script(
    *,
    python: str,
    packages: Sequence[str],
    requirements: Sequence[str],
    config: str,
    ccache: bool,
    owner: str | None,
    paths: Sequence[str],
    quiet: bool = False,
) -> str:
    """Return the shell script the container runs.

    Parameters
    ----------
    python : str
        Interpreter inside the image.

    packages : Sequence[str]
        System packages to install with ``dnf`` or ``yum``.

    requirements : Sequence[str]
        Python packages PyVista needs at run time.

    config : str
        JSON build configuration passed to ``python -m pyvista._build``.

    ccache : bool
        Whether to install a static ``ccache``.

    owner : str or None
        ``uid:gid`` to hand the written files back to, or ``None`` to leave them.

    paths : Sequence[str]
        Mounted directories to hand back.

    quiet : bool, default: False
        Install system packages without listing each one.

    Returns
    -------
    str
        Bash script.

    """
    lines = ['set -euo pipefail']
    if owner is not None and paths:
        lines.append(f"trap 'chown -R {owner} {' '.join(paths)} || true' EXIT")
    lines += [
        f'PY={shlex.quote(python)}',
        'if ! test -x "$PY"; then echo "$PY is not in this image" >&2; exit 1; fi',
        "git config --global --add safe.directory '*'",
        f'export PIP_CACHE_DIR={PIP_CACHE}',
    ]
    if packages:
        pkgs = ' '.join(shlex.quote(p) for p in packages)
        flags = '-y -q' if quiet else '-y'
        lines.append(
            'if command -v dnf >/dev/null; then '
            f'dnf install {flags} --setopt=install_weak_deps=False {pkgs}; '
            f'else yum install {flags} {pkgs}; fi'
        )
    if ccache:
        name = f'ccache-{CCACHE_VERSION}-linux-$ARCH-musl-static'
        url = f'https://github.com/ccache/ccache/releases/download/v{CCACHE_VERSION}/{name}.tar.xz'
        cases = ' '.join(f'{arch}) SHA={sha};;' for arch, sha in CCACHE_SHA256.items())
        lines += [
            'ARCH=$(uname -m)',
            f'case "$ARCH" in {cases} *) echo "no static ccache for $ARCH" >&2; exit 1;; esac',
            f'curl -sSfL -o /tmp/ccache.tar.xz "{url}"',
            'echo "$SHA  /tmp/ccache.tar.xz" | sha256sum --check --quiet',
            f'tar -xJf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 "{name}/ccache"',
        ]
    reqs = ' '.join(shlex.quote(r) for r in requirements)
    lines += [
        f'"$PY" -m pip install --quiet --root-user-action=ignore --only-binary :all: {reqs}',
        f'PYTHONPATH={PYVISTA_ROOT} "$PY" -m pyvista._build {shlex.quote(config)}',
    ]
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
        Container path to host directory. The PyVista package is mounted read-only.

    script : str
        Script passed to ``bash -c`` inside the container.

    interactive : bool
        Attach a terminal so the build inside can ask questions.

    env : Mapping[str, str]
        Host environment; its non-path ``CCACHE_*`` and ``SCCACHE_*`` settings pass through.

    Returns
    -------
    list[str]
        Command and arguments.

    """
    cmd = ['docker', 'run', '--rm', '--init', '-i']
    if interactive:
        cmd.append('-t')
    for inside, outside in mounts.items():
        mode = ':ro' if inside.startswith(PYVISTA_ROOT) else ''
        cmd += ['--volume', f'{outside}:{inside}{mode}']
    for name in sorted(env):
        if name.startswith(('CCACHE_', 'SCCACHE_')) and name not in _HOST_PATH_VARS:
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
