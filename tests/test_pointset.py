"""Test pyvista.PointSet"""

import numpy as np
import pytest

import pyvista

# skip all tests if concrete pointset unavailable
pytestmark = pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0),
    reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)


def test_pointset_basic():
    # create empty pointset
    pset = pyvista.PointSet()
    assert pset.n_points == 0
    assert pset.n_cells == 0
    assert 'PointSet' in str(pset)


def test_pointset(pointset):
    assert pointset.n_points == pointset.points.shape[0]
    assert pointset.n_cells == 0

    arr_name = 'arr'
    pointset.point_data[arr_name] = np.random.random(10)
    assert arr_name in pointset.point_data
