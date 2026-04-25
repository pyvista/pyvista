from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING

import pytest

from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_warn_external(recwarn: pytest.WarningsRecorder):
    """Taken and adapted from
    https://github.com/matplotlib/matplotlib/blob/a00d606d592bcf8d335f4f3ac2768882d3a49e7b/lib/matplotlib/tests/test_cbook.py#L509"""

    warn_external('oops')
    assert len(recwarn) == 1
    if sys.version_info[:2] >= (3, 12):
        # With Python 3.12, we let Python figure out the stacklevel using the
        # `skip_file_prefixes` argument, which cannot exempt tests, so just confirm
        # the filename is not in the package.
        basedir = pathlib.Path(__file__).parents[1]
        assert not recwarn[0].filename.startswith(str(basedir / 'pyvista'))

    else:
        # On older Python versions, we manually calculated the stacklevel, and had an
        # exception for our own tests.
        assert recwarn[0].filename == __file__


def test_warn_external_frame_embedded_python(mocker: MockerFixture):
    """Taken and adapted from
    https://github.com/matplotlib/matplotlib/blob/a00d606d592bcf8d335f4f3ac2768882d3a49e7b/lib/matplotlib/tests/test_cbook.py#L525"""
    m = mocker.patch.object(sys, '_getframe')
    m.return_value = None
    with pytest.warns(UserWarning, match=r'\Adummy\Z'):
        warn_external('dummy')
