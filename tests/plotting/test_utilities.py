from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting import _vtk
from pyvista.plotting.utilities import algorithms
from pyvista.plotting.utilities import xvfb

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.skip_windows
@pytest.mark.skip_mac
def test_start_xvfb():
    def _test_start_xvfb():
        pv.start_xvfb()
        if pv._version.version_info[:2] > (0, 48):
            msg = 'Remove this method'
            raise RuntimeError(msg)

    with pytest.warns(
        PyVistaDeprecationWarning,
        match='This function is deprecated and will be removed in future version',
    ):
        _test_start_xvfb()


def test_start_xvfb_raises(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    monkeypatch.setattr(os, 'name', 'foo')
    with (
        pytest.raises(OSError, match='`start_xvfb` is only supported on Linux'),
        pytest.warns(
            PyVistaDeprecationWarning,
            match='This function is deprecated and will be removed in future version',
        ),
    ):
        pv.start_xvfb()

    monkeypatch.setattr(os, 'name', 'posix')

    m = mocker.patch.object(os, 'system')
    m.return_value = True

    with (
        pytest.raises(OSError, match=re.escape(xvfb.XVFB_INSTALL_NOTES)),
        pytest.warns(
            PyVistaDeprecationWarning,
            match='This function is deprecated and will be removed in future version',
        ),
    ):
        pv.start_xvfb()


def test_algo_to_mesh_handler_raises(mocker: MockerFixture):
    m = mocker.patch.object(algorithms, 'wrap')
    m.return_value = None

    with pytest.raises(
        pv.PyVistaPipelineError, match=r'The passed algorithm is failing to produce an output.'
    ):
        algorithms.algorithm_to_mesh_handler(_vtk.vtkAlgorithm())
