import vtk
import pytest
import vtki
from vtki import examples
import numpy as np

grid = vtki.UnstructuredGrid(examples.hexbeamfile)


def test_point_arrays():
    key = 'test_array'
    grid.point_arrays[key] = np.arange(grid.number_of_points)
    assert key in grid.point_arrays

    orig_value = grid.point_arrays[key][0]/1.0
    grid.point_arrays[key][0] += 1
    assert orig_value == grid._point_scalar(key)[0] -1

    del grid.point_arrays[key]
    assert key not in grid.point_arrays

    grid.point_arrays[key] = np.arange(grid.number_of_points)
    assert key in grid.point_arrays


def test_point_arrays_bad_value():
    with pytest.raises(TypeError):
        grid.point_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.point_arrays['new_array'] = np.arange(grid.number_of_points - 1)


def test_cell_arrays():
    key = 'test_array'
    grid.cell_arrays[key] = np.arange(grid.number_of_cells)
    assert key in grid.cell_arrays

    orig_value = grid.cell_arrays[key][0]/1.0
    grid.cell_arrays[key][0] += 1
    assert orig_value == grid.cell_arrays[key][0] -1

    del grid.cell_arrays[key]
    assert key not in grid.cell_arrays

    grid.cell_arrays[key] = np.arange(grid.number_of_cells)
    assert key in grid.cell_arrays


def test_cell_arrays_bad_value():
    with pytest.raises(TypeError):
        grid.cell_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.cell_arrays['new_array'] = np.arange(grid.number_of_cells - 1)


def test_copy():
    grid_copy = grid.copy(deep=True)
    grid_copy.points[0] = np.nan
    assert not np.any(np.isnan(grid.points[0]))

    grid_copy_shallow = grid.copy(deep=False)
    grid_copy.points[0] += 0.1
    assert np.all(grid_copy_shallow.points[0] == grid.points[0])


def test_transform():
    trans = vtk.vtkTransform()
    trans.RotateX(30)
    trans.RotateY(30)
    trans.RotateZ(30)
    trans.Translate(1, 1, 2)
    trans.Update()

    grid_a = grid.copy()
    grid_b = grid.copy()
    grid_c = grid.copy()
    grid_a.transform(trans)
    grid_b.transform(trans.GetMatrix())
    grid_c.transform(vtki.trans_from_matrix(trans.GetMatrix()))
    assert np.allclose(grid_a.points, grid_b.points)
    assert np.allclose(grid_a.points, grid_c.points)


def test_transform_errors():
    with pytest.raises(TypeError):
        grid.transform(None)

    with pytest.raises(Exception):
        grid.transform(np.array([1]))


def test_translate():
    grid_copy = grid.copy()
    xyz = [1, 1, 1]
    grid_copy.translate(xyz)

    grid_points = grid.points.copy() + np.array(xyz)
    assert np.allclose(grid_copy.points, grid_points)


def test_rotate_x():
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateX(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = vtki.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_x(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_rotate_y():
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateY(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = vtki.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_y(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_rotate_z():
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateZ(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = vtki.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_z(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_make_points_double():
    grid_copy = grid.copy()
    grid_copy.points = grid_copy.points.astype(np.float32)
    assert grid_copy.points.dtype == np.float32
    grid_copy.points_to_double()
    assert grid_copy.points.dtype == np.double


def test_invalid_points():
    with pytest.raises(TypeError):
        grid.points = None


def test_points_bool():
    bool_arr = np.zeros(grid.number_of_points, np.bool)
    grid.point_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.point_arrays['bool_arr'].all()
    assert grid._point_scalar('bool_arr').all()
    assert grid._point_scalar('bool_arr').dtype == np.bool


def test_cells_bool():
    bool_arr = np.zeros(grid.number_of_cells, np.bool)
    grid.cell_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.cell_arrays['bool_arr'].all()
    assert grid._cell_scalar('bool_arr').all()
    assert grid._cell_scalar('bool_arr').dtype == np.bool
