from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from pyvista.plotting import _vtk
from pyvista.plotting.utilities import algorithms

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_start_xvfb_removed():
    assert not hasattr(pv, 'start_xvfb')


def test_algo_to_mesh_handler_raises(mocker: MockerFixture):
    m = mocker.patch.object(algorithms, 'wrap')
    m.return_value = None

    with pytest.raises(
        pv.PyVistaPipelineError, match=r'The passed algorithm is failing to produce an output.'
    ):
        algorithms.algorithm_to_mesh_handler(_vtk.vtkAlgorithm())
