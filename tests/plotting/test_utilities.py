from __future__ import annotations

import os
import platform
import re
from typing import TYPE_CHECKING

import pytest
import vtk

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting.utilities import algorithms
from pyvista.plotting.utilities import xvfb

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

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
    if pv._version.version_info[:2] > (0, 48):
        msg = 'Remove this method'
        raise RuntimeError(msg)


def test_start_xvfb_raises(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    monkeypatch.setattr(os, 'name', 'foo')
    with (
        pytest.raises(OSError, match='`start_xvfb` is only supported on Linux'),
        pytest.warns(PyVistaDeprecationWarning),
    ):
        pv.start_xvfb()

    monkeypatch.setattr(os, 'name', 'posix')

    m = mocker.patch.object(os, 'system')
    m.return_value = True

    with (
        pytest.raises(OSError, match=re.escape(xvfb.XVFB_INSTALL_NOTES)),
        pytest.warns(PyVistaDeprecationWarning),
    ):
        pv.start_xvfb()


def test_algo_to_mesh_handler_raises(mocker: MockerFixture):
    m = mocker.patch.object(algorithms, 'wrap')
    m.return_value = None

    with pytest.raises(
        pv.PyVistaPipelineError, match='The passed algorithm is failing to produce an output.'
    ):
        algorithms.algorithm_to_mesh_handler(vtk.vtkAlgorithm())
