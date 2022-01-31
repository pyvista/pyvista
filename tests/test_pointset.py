"""Test pyvista.PointSet"""

import numpy as np
import pytest

import pyvista

# skip all tests concrete pointset unavailable
if pyvista.vtk_version_info < (9, 1, 0):
    pytestmark = pytest.mark.skip


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
