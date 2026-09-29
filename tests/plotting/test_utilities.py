from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pyvista as pv
from pyvista import _vtk
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
        algorithms.algorithm_to_mesh_handler(_vtk.vtkSphereSource())


@pytest.mark.parametrize('port', [1, -1])
def test_algo_to_mesh_handler_port_out_of_range_raises(port):
    match = rf'Port {port} is out of range for vtkSphereSource, which has 1 output port\(s\).'
    with pytest.raises(pv.PyVistaPipelineError, match=match):
        algorithms.algorithm_to_mesh_handler(_vtk.vtkSphereSource(), port=port)


def test_algo_to_mesh_handler_no_output_port_raises():
    match = r'Port 0 is out of range for vtkAlgorithm, which has 0 output port\(s\).'
    with pytest.raises(pv.PyVistaPipelineError, match=match):
        algorithms.algorithm_to_mesh_handler(_vtk.vtkAlgorithm())
