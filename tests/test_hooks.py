"""Test the pre-commit hooks"""

from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope='session')
def pre_commit_config(request: pytest.FixtureRequest):
    with (request.config.rootpath / '.pre-commit-config.yaml').open() as f:
        return yaml.safe_load(f)


def test_warnings_converter(
    tmp_path: Path,
    pre_commit_config: dict,
    request: pytest.FixtureRequest,
):
    test = """\
    import warnings

    warnings.warn("foo")
    warnings.warn("foo", UserWarning)
    warnings.warn("foo", UserWarning, 1)
    warnings.warn("foo", UserWarning, stacklevel=1)
    warnings.warn("foo", category=UserWarning, stacklevel=1)
    warnings.warn(message="foo", category=UserWarning, stacklevel=1)
    warnings.warn(category=UserWarning, stacklevel=1, message="foo")
    """

    if sys.version_info[:2] >= (3, 12):
        test += """
    warnings.warn(category=UserWarning, stacklevel=1, message="foo", source='bar', skip_file_prefixes=('',))
    """  # noqa: E501

    with (file := (tmp_path / 'file.py')).open('w') as f:
        f.write(textwrap.dedent(test))

    local = next(v for v in pre_commit_config['repos'] if v['repo'] == 'local')
    warning_hook = next(v for v in local['hooks'] if v['id'] == 'warn_external')
    cml = warning_hook['entry']

    ret = subprocess.run(
        [sys.executable, *shlex.split(cml)[1:], str(file.absolute())],
        check=True,
        cwd=request.config.rootpath,
    )
    assert ret.returncode == 0

    with file.open('r') as f:
        lines = f.readlines()

    expected = """\
        from pyvista._warn_external import warn_external

        warn_external("foo")
        warn_external("foo", UserWarning)
        warn_external("foo", UserWarning)
        warn_external("foo", UserWarning)
        warn_external("foo", category=UserWarning)
        warn_external(message="foo", category=UserWarning)
        warn_external(message="foo", category=UserWarning)
        """

    if sys.version_info[:2] >= (3, 12):
        expected += """
        warn_external(message="foo", category=UserWarning)
        """

    assert textwrap.dedent(expected) == ''.join(lines)
