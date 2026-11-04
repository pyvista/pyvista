"""Fail unless the installed vtk is the one a `vtk_local`, `vtk_dev` or `vtk_latest` env asked for.

Standalone rather than part of toxfile.py (a tox plugin -- tox/tox_uv aren't installed
inside a testenv) so this can run as a commands_pre step, after the install that step makes.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen

from packaging.utils import parse_wheel_filename
from packaging.version import Version


def _installed() -> Version | None:
    """Return the installed vtk version, or None when vtk is not installed."""
    try:
        return Version(installed_version('vtk'))
    except PackageNotFoundError:
        return None


def check_local(installed: Version | None) -> None:
    """Require the version of the wheel in VTK_WHEEL_DIR."""
    wheel_dir = Path(os.environ['VTK_WHEEL_DIR'])
    wheels = list(wheel_dir.glob('vtk-*.whl'))
    if not wheels:
        msg = f'No VTK wheel found in VTK_WHEEL_DIR ({wheel_dir}).'
        raise SystemExit(msg)

    _name, expected, _build, _tags = parse_wheel_filename(wheels[0].name)
    if installed != expected:
        msg = (
            f'Expected the local VTK wheel ({expected}) to be installed, but found'
            f' {installed} instead.'
        )
        raise SystemExit(msg)


def check_dev(installed: Version | None) -> None:
    """Require a development build."""
    if installed is None or not installed.is_devrelease:
        msg = f'Expected a VTK development wheel to be installed, but found {installed} instead.'
        raise SystemExit(msg)


def check_latest(installed: Version | None) -> None:
    """Require at least the newest VTK release on PyPI."""
    with urlopen('https://pypi.org/pypi/vtk/json', timeout=30) as response:
        expected = Version(json.load(response)['info']['version'])
    if installed is None or installed < expected:
        msg = (
            f'Expected the latest VTK release ({expected}) or newer to be installed, but found'
            f' {installed} instead.'
        )
        raise SystemExit(msg)


CHECKS = {'local': check_local, 'dev': check_dev, 'latest': check_latest}


def main() -> None:
    """Run the check named by the first argument."""
    CHECKS[sys.argv[1]](_installed())


if __name__ == '__main__':
    main()
