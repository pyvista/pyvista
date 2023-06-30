"""Test pyvista.PointSet"""

import numpy as np
import pytest
import vtk

import pyvista
from pyvista.core.errors import (
    PointSetCellOperationError,
    PointSetDimensionReductionError,
    PointSetNotSupported,
)

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
    assert pset.area == 0
    assert pset.volume == 0


def test_pointset_from_vtk():
    vtk_pset = vtk.vtkPointSet()

    np_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = pyvista.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pyvista.PointSet(vtk_pset, deep=False)
    assert pset.n_points == 2

    # test that data is shallow copied

    np_points[:] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert np.array_equal(np_points, pset.points)

    # test that data is deep copied

    np_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    point_copy = np_points.copy()
    points = pyvista.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pyvista.PointSet(vtk_pset, deep=True)

    np_points[:] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not np.array_equal(np_points, pset.points)
    assert np.array_equal(pset.points, point_copy)


def test_pointset_wrap():
    vtk_pset = vtk.vtkPointSet()
    np_points = np.array([[0.0, 0.0, 0.0]])
    points = pyvista.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pyvista.wrap(vtk_pset)
    assert type(pset) is pyvista.PointSet

    # test that wrapping is shallow copied
    pset.points[:] = np.array([[1.0, 0.0, 0.0]])
    assert np.array_equal(vtk_pset.GetPoint(0), pset.points[0])


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


@pytest.mark.parametrize("force_float,expected_data_type", [(False, np.int64), (True, np.float32)])
def test_pointset_force_float(force_float, expected_data_type):
    np_points = np.array([[1, 2, 3]], np.int64)
    if force_float:
        with pytest.warns(UserWarning, match='Points is not a float type'):
            pset = pyvista.PointSet(np_points, force_float=force_float)
    else:
        pset = pyvista.PointSet(np_points, force_float=force_float)
    assert pset.points.dtype == expected_data_type


def test_center_of_mass():
    np_points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    pset = pyvista.PointSet(np_points)
    assert np.allclose(pset.center_of_mass(), [0.5, 0.0, 0.5])


def test_points_to_double():
    np_points = np.array([[1, 2, 3]], np.int64)
    pset = pyvista.PointSet(np_points, force_float=False)
    assert pset.points_to_double().points.dtype == np.double


def test_translate():
    np_points = np.array([1, 2, 3], np.int64)
    with pytest.warns(UserWarning, match='Points is not a float type'):
        pset = pyvista.PointSet(np_points)
    pset.translate((4, 3, 2), inplace=True)
    assert np.allclose(pset.center, [5, 5, 5])


def test_scale():
    np_points = np.array([1, 2, 3], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.scale(2, inplace=True)
    assert np.allclose(pset.points, [2.0, 4.0, 6.0])


def test_flip_x():
    np_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.flip_x(inplace=True)
    assert np.allclose(
        pset.points,
        np.array(
            [
                [7.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 8.0, 9.0],
            ],
        ),
    )


def test_flip_y():
    np_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.flip_y(inplace=True)
    assert np.allclose(
        pset.points,
        np.array(
            [
                [1.0, 8.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 2.0, 9.0],
            ],
        ),
    )


def test_flip_z():
    np_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.flip_z(inplace=True)
    assert np.allclose(
        pset.points,
        np.array(
            [
                [1.0, 2.0, 9.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 3.0],
            ],
        ),
    )


def test_flip_normal():
    np_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.flip_normal([1.0, 1.0, 1.0], inplace=True)
    assert np.allclose(
        pset.points,
        np.array(
            [
                [7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
            ],
        ),
    )


def test_threshold(pointset):
    pointset['scalars'] = range(pointset.n_points)
    out = pointset.threshold(pointset.n_points // 2)
    assert isinstance(out, pyvista.PointSet)
    assert out.n_points == pointset.n_points // 2


def test_threshold_percent(pointset):
    pointset['scalars'] = range(pointset.n_points)
    out = pointset.threshold_percent(50)
    assert isinstance(out, pyvista.PointSet)
    assert out.n_points == pointset.n_points // 2


def test_explode(pointset):
    out = pointset.explode(1)
    assert isinstance(out, pyvista.PointSet)
    ori_xlen = pointset.bounds[1] - pointset.bounds[0]
    new_xlen = out.bounds[1] - out.bounds[0]
    assert np.isclose(2 * ori_xlen, new_xlen)


def test_delaunay_3d(pointset):
    out = pointset.delaunay_3d()
    assert isinstance(out, pyvista.UnstructuredGrid)
    assert out.n_cells > 10


def test_raise_unsupported(pointset):
    with pytest.raises(PointSetNotSupported):
        pointset.contour()

    with pytest.raises(PointSetNotSupported):
        pointset.cell_data_to_point_data()

    with pytest.raises(PointSetNotSupported):
        pointset.point_data_to_cell_data()

    with pytest.raises(PointSetCellOperationError):
        pointset.triangulate()

    with pytest.raises(PointSetCellOperationError):
        pointset.decimate_boundary()

    with pytest.raises(PointSetCellOperationError):
        pointset.find_cells_along_line()

    with pytest.raises(PointSetCellOperationError):
        pointset.tessellate()

    with pytest.raises(PointSetDimensionReductionError):
        pointset.slice()

    with pytest.raises(PointSetDimensionReductionError):
        pointset.slice_along_axis()

    with pytest.raises(PointSetDimensionReductionError):
        pointset.slice_along_line()

    with pytest.raises(PointSetDimensionReductionError):
        pointset.slice_implicit()

    with pytest.raises(PointSetDimensionReductionError):
        pointset.slice_orthogonal()

    with pytest.raises(PointSetCellOperationError):
        pointset.shrink()

    with pytest.raises(PointSetCellOperationError):
        pointset.separate_cells()

    with pytest.raises(PointSetCellOperationError):
        pointset.remove_cells()

    with pytest.raises(PointSetCellOperationError):
        pointset.point_is_inside_cell()


def test_rotate_x():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.rotate_x(45, inplace=True)
    assert np.allclose(pset.points, [1.0, 0.0, 1.4142135])


def test_rotate_y():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.rotate_y(45, inplace=True)
    assert np.allclose(pset.points, [1.4142135, 1.0, 0.0])


def test_rotate_z():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.rotate_z(45, inplace=True)
    assert np.allclose(pset.points, [0.0, 1.4142135, 1.0])


def test_rotate_vector():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pyvista.PointSet(np_points)
    pset.rotate_vector([1, 2, 1], 45, inplace=True)
    assert np.allclose(pset.points, [1.1910441, 1.0976311, 0.6136938])
