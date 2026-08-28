"""Fail unless the installed vtk matches the wheel in VTK_WHEEL_DIR.

Standalone rather than part of toxfile.py (a tox plugin -- tox/tox_uv aren't installed
inside a testenv) so this can run as a `vtk_local` commands_pre step.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
import os
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version


def main() -> None:
    """Compare the installed vtk version against the wheel in VTK_WHEEL_DIR."""
    wheel_dir = Path(os.environ['VTK_WHEEL_DIR'])
    wheels = list(wheel_dir.glob('vtk-*.whl'))
    if not wheels:
        msg = f'No VTK wheel found in VTK_WHEEL_DIR ({wheel_dir}).'
        raise SystemExit(msg)

    _name, expected, _build, _tags = parse_wheel_filename(wheels[0].name)
    try:
        installed = Version(installed_version('vtk'))
    except PackageNotFoundError:
        installed = None

    if installed != expected:
        msg = (
            f'Expected the local VTK wheel ({expected}) to be installed, but found'
            f' {installed} instead.'
        )
        raise SystemExit(msg)


if __name__ == '__main__':
    main()
