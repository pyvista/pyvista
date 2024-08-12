from __future__ import annotations

import pytest

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning


def test_start_xvfb():
    with pytest.warns(PyVistaDeprecationWarning):
        pv.start_xvfb()
    if pv._version.version_info >= (0, 48):
        raise RuntimeError("Remove this method")
