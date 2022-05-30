"""Test pyvista.PointSet"""

import numpy as np
import pytest

import pyvista

# skip all tests if concrete pointset unavailable
pytestmark = pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)


def test_pointset_basic():
    # create empty pointset
    pset = pyvista.PointSet()
    assert pset.n_points == 0
    assert pset.n_cells == 0
    assert 'PointSet' in str(pset)
    assert 'PointSet' in repr(pset)


def test_pointset(pointset):
    assert pointset.n_points == pointset.points.shape[0]
    assert pointset.n_cells == 0

    arr_name = 'arr'
    pointset.point_data[arr_name] = np.random.random(10)
    assert arr_name in pointset.point_data

    # test that points can be modified
    pointset.points[:] = 0
    assert np.allclose(pointset.points, 0)
    pointset.points = np.ones((10, 3))
    assert np.allclose(pointset.points, 1)


def test_save(tmpdir, pointset):
    filename = str(tmpdir.mkdir("tmpdir").join(f'{"tmp.xyz"}'))
    pointset.save(filename)
    points = np.loadtxt(filename)
    assert np.allclose(points, pointset.points)


@pytest.mark.parametrize('deep', [True, False])
def test_cast_to_polydata(pointset, deep):
    data = np.linspace(0, 1, pointset.n_points)
    key = 'key'
    pointset.point_data[key] = data

    pdata = pointset.cast_to_polydata(deep)
    assert key in pdata.point_data
    assert np.allclose(pdata.point_data[key], pointset.point_data[key])
    pdata.point_data[key][:] = 0
    if deep:
        assert not np.allclose(pdata.point_data[key], pointset.point_data[key])
    else:
        assert np.allclose(pdata.point_data[key], pointset.point_data[key])


def test_filters_return_pointset(sphere):
    pointset = sphere.cast_to_pointset()
    clipped = pointset.clip()
    assert isinstance(clipped, pyvista.PointSet)
