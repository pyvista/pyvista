"""Test pyvista.PointSet"""

from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import PointSetCellOperationError
from pyvista.core.errors import PointSetDimensionReductionError
from pyvista.core.errors import PointSetNotSupported

# skip all tests if concrete pointset unavailable
pytestmark = pytest.mark.needs_vtk_version(
    9, 1, 0, reason='Requires VTK>=9.1.0 for a concrete PointSet class'
)


def test_pointset_basic():
    # create empty pointset
    pset = pv.PointSet()
    assert pset.n_points == 0
    assert pset.n_cells == 0
    assert 'PointSet' in str(pset)
    assert 'PointSet' in repr(pset)
    assert pset.area == 0
    assert pset.volume == 0


def test_pointset_from_vtk():
    vtk_pset = vtk.vtkPointSet()

    np_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = pv.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pv.PointSet(vtk_pset, deep=False)
    assert pset.n_points == 2

    # test that data is shallow copied

    np_points[:] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert np.array_equal(np_points, pset.points)

    # test that data is deep copied

    np_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    point_copy = np_points.copy()
    points = pv.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pv.PointSet(vtk_pset, deep=True)

    np_points[:] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not np.array_equal(np_points, pset.points)
    assert np.array_equal(pset.points, point_copy)


def test_pointset_wrap():
    vtk_pset = vtk.vtkPointSet()
    np_points = np.array([[0.0, 0.0, 0.0]])
    points = pv.vtk_points(np_points, deep=False)
    vtk_pset.SetPoints(points)

    pset = pv.wrap(vtk_pset)
    assert type(pset) is pv.PointSet

    # test that wrapping is shallow copied
    pset.points[:] = np.array([[1.0, 0.0, 0.0]])
    assert np.array_equal(vtk_pset.GetPoint(0), pset.points[0])


def test_pointset(pointset):
    assert pointset.n_points == pointset.points.shape[0]
    assert pointset.n_cells == 0

    arr_name = 'arr'
    pointset.point_data[arr_name] = np.random.default_rng().random(10)
    assert arr_name in pointset.point_data

    # test that points can be modified
    pointset.points[:] = 0
    assert np.allclose(pointset.points, 0)
    pointset.points = np.ones((10, 3))
    assert np.allclose(pointset.points, 1)


def test_save(tmpdir, pointset):
    filename = str(tmpdir.mkdir('tmpdir').join(f'{"tmp.xyz"}'))
    pointset.save(filename)
    points = np.loadtxt(filename)
    assert np.allclose(points, pointset.points)


@pytest.mark.parametrize('deep', [True, False])
def test_cast_to_polydata(pointset, deep):
    data = np.linspace(0, 1, pointset.n_points)
    key = 'key'
    pointset.point_data[key] = data

    pdata = pointset.cast_to_polydata(deep=deep)
    assert isinstance(pdata, pv.PolyData)
    assert key in pdata.point_data
    assert np.allclose(pdata.point_data[key], pointset.point_data[key])
    pdata.point_data[key][:] = 0
    if deep:
        assert not np.allclose(pdata.point_data[key], pointset.point_data[key])
    else:
        assert np.allclose(pdata.point_data[key], pointset.point_data[key])


def test_cast_to_unstructured_grid(pointset):
    data = np.linspace(0, 1, pointset.n_points)
    key = 'key'
    pointset.point_data[key] = data

    pdata = pointset.cast_to_unstructured_grid()
    assert isinstance(pdata, pv.UnstructuredGrid)
    assert key in pdata.point_data
    assert np.allclose(pdata.point_data[key], pointset.point_data[key])
    pdata.point_data[key][:] = 0
    assert not np.allclose(pdata.point_data[key], pointset.point_data[key])


def test_filters_return_pointset(sphere):
    pointset = sphere.cast_to_pointset()
    clipped = pointset.clip()
    assert isinstance(clipped, pv.PointSet)


def test_pointset_clip_vtk_bug(sphere):
    pointset = sphere.cast_to_pointset()
    alg = vtk.vtkTableBasedClipDataSet()
    alg.SetClipFunction(pv.generate_plane((1, 0, 0), (0, 0, 0)))

    # Filter works with PolyData
    alg.SetInputData(sphere)
    alg.Update()
    out = pv.wrap(alg.GetOutput())
    assert not out.is_empty

    # Bug: filter returns empty mesh with PointSet
    alg.SetInputData(pointset)
    alg.Update()
    out = pv.wrap(alg.GetOutput())
    if pv.vtk_version_info >= (9, 4) and pv.vtk_version_info < (9, 5):
        # A vtk bug was introduced in 9.4 https://gitlab.kitware.com/vtk/vtk/-/issues/19649
        # Which has been fixed for vtk 9.5: https://gitlab.kitware.com/vtk/vtk/-/merge_requests/12040
        assert out.is_empty
    else:
        assert not out.is_empty


@pytest.mark.parametrize(
    ('force_float', 'expected_data_type'),
    [(False, np.int64), (True, np.float32)],
)
def test_pointset_force_float(force_float, expected_data_type):
    np_points = np.array([[1, 2, 3]], np.int64)
    if force_float:
        with pytest.warns(UserWarning, match='Points is not a float type'):
            pset = pv.PointSet(np_points, force_float=force_float)
    else:
        pset = pv.PointSet(np_points, force_float=force_float)
    assert pset.points.dtype == expected_data_type


def test_center_of_mass():
    np_points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    pset = pv.PointSet(np_points)
    assert np.allclose(pset.center_of_mass(), [0.5, 0.0, 0.5])


def test_points_to_double():
    np_points = np.array([[1, 2, 3]], np.int64)
    pset = pv.PointSet(np_points, force_float=False)
    assert pset.points_to_double().points.dtype == np.double


def test_translate():
    np_points = np.array([1, 2, 3], np.int64)
    with pytest.warns(UserWarning, match='Points is not a float type'):
        pset = pv.PointSet(np_points)
    pset.translate((4, 3, 2), inplace=True)
    assert np.allclose(pset.center, [5, 5, 5])


def test_scale():
    np_points = np.array([1, 2, 3], dtype=float)
    pset = pv.PointSet(np_points)
    pset.scale(2, inplace=True)
    assert np.allclose(pset.points, [2.0, 4.0, 6.0])


def test_flip_x():
    np_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    pset = pv.PointSet(np_points)
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
    pset = pv.PointSet(np_points)
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
    pset = pv.PointSet(np_points)
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
    pset = pv.PointSet(np_points)
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
    assert isinstance(out, pv.PointSet)
    assert out.n_points == pointset.n_points // 2


def test_threshold_percent(pointset):
    pointset['scalars'] = range(pointset.n_points)
    out = pointset.threshold_percent(50)
    assert isinstance(out, pv.PointSet)
    assert out.n_points == pointset.n_points // 2


def test_explode(pointset):
    out = pointset.explode(1)
    assert isinstance(out, pv.PointSet)
    ori_xlen = pointset.bounds.x_max - pointset.bounds.x_min
    new_xlen = out.bounds.x_max - out.bounds.x_min
    assert np.isclose(2 * ori_xlen, new_xlen)


def test_delaunay_3d(pointset):
    out = pointset.delaunay_3d()
    assert isinstance(out, pv.UnstructuredGrid)
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

    with pytest.raises(PointSetCellOperationError):
        pointset.extract_surface()

    with pytest.raises(PointSetCellOperationError):
        pointset.extract_geometry()


def test_rotate_x():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pv.PointSet(np_points)
    pset.rotate_x(45, inplace=True)
    assert np.allclose(pset.points, [1.0, 0.0, 1.4142135])


def test_rotate_y():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pv.PointSet(np_points)
    pset.rotate_y(45, inplace=True)
    assert np.allclose(pset.points, [1.4142135, 1.0, 0.0])


def test_rotate_z():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pv.PointSet(np_points)
    pset.rotate_z(45, inplace=True)
    assert np.allclose(pset.points, [0.0, 1.4142135, 1.0])


def test_rotate_vector():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pv.PointSet(np_points)
    pset.rotate_vector([1, 2, 1], 45, inplace=True)
    assert np.allclose(pset.points, [1.1910441, 1.0976311, 0.6136938])


def test_rotate():
    np_points = np.array([1, 1, 1], dtype=float)
    pset = pv.PointSet(np_points)
    pset.rotate([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], inplace=True)
    assert np.allclose(pset.points, [-1, -1, -1])


@pytest.mark.parametrize(
    ('grid_class', 'dimensionality', 'dimensions'),
    [
        (pv.ExplicitStructuredGrid, 3, (2, 42, 142)),
        (pv.StructuredGrid, 0, (1, 1, 1)),
        (pv.StructuredGrid, 1, (1, 42, 1)),
        (pv.StructuredGrid, 2, (42, 1, 142)),
        (pv.StructuredGrid, 3, (2, 42, 142)),
    ],
)
def test_pointgrid_dimensionality(grid_class, dimensionality, dimensions):
    if grid_class == pv.ExplicitStructuredGrid:
        # ExplicitStructuredGrid only supports 3D
        grid = pv.examples.load_explicit_structured(dimensions=dimensions)
    elif grid_class == pv.StructuredGrid:
        x, y, z = np.meshgrid(
            np.arange(dimensions[0], dtype=np.float32),
            np.arange(dimensions[1], dtype=np.float32),
            np.arange(dimensions[2], dtype=np.float32),
            indexing='ij',
        )
        grid = grid_class(x, y, z)

    assert grid.dimensionality == dimensionality
    assert grid.dimensionality == grid.get_cell(0).GetCellDimension()


@pytest.mark.parametrize(
    ('attr', 'mesh', 'expected'),
    [
        (
            'polyhedron_faces',
            examples.cells.Polyhedron(),
            [3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3],
        ),
        ('polyhedron_face_locations', examples.cells.Polyhedron(), [4, 0, 1, 2, 3]),
        ('polyhedron_faces', pv.UnstructuredGrid(), []),
        ('polyhedron_face_locations', pv.UnstructuredGrid(), []),
    ],
)
def test_polyhedron_faces_and_face_locations(attr, mesh, expected):
    actual = getattr(mesh, attr)
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == int
    assert np.array_equal(actual, expected)

    if pv.vtk_version_info >= (9, 4) and pv.vtk_version_info <= (9, 5, 0):
        # Deprecated in 9.4, removed in 9.6
        with pytest.warns(DeprecationWarning, match=r'Call to deprecated method'):
            # Test deprecation warning is emitted by VTK
            getattr(mesh, attr.split('polyhedron_')[1])
