from __future__ import annotations

import os
import platform

import pytest

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning

skip_windows = pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')
skip_mac = pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason='MacOS CI fails when downloading examples',
)


@skip_windows
@skip_mac
def test_start_xvfb():
    with pytest.warns(PyVistaDeprecationWarning):
        pv.start_xvfb()
    if pv._version.version_info >= (0, 48):
        raise RuntimeError('Remove this method')
