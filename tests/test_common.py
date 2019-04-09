import sys

import numpy as np
import pytest
import vtk

import vtki
from vtki import examples

grid = vtki.UnstructuredGrid(examples.hexbeamfile)

py2 = sys.version_info.major == 2


def test_point_arrays():
    key = 'test_array'
    grid[key] = np.arange(grid.n_points)
    assert key in grid.point_arrays

    orig_value = grid.point_arrays[key][0]/1.0
    grid.point_arrays[key][0] += 1
    assert orig_value == grid._point_scalar(key)[0] -1

    del grid.point_arrays[key]
    assert key not in grid.point_arrays

    grid.point_arrays[key] = np.arange(grid.n_points)
    assert key in grid.point_arrays

    assert np.allclose(grid[key], np.arange(grid.n_points))


def test_point_arrays_bad_value():
    with pytest.raises(TypeError):
        grid.point_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.point_arrays['new_array'] = np.arange(grid.n_points - 1)


def test_cell_arrays():
    key = 'test_array'
    grid[key] = np.arange(grid.n_cells)
    assert key in grid.cell_arrays

    orig_value = grid.cell_arrays[key][0]/1.0
    grid.cell_arrays[key][0] += 1
    assert orig_value == grid.cell_arrays[key][0] -1

    del grid.cell_arrays[key]
    assert key not in grid.cell_arrays

    grid.cell_arrays[key] = np.arange(grid.n_cells)
    assert key in grid.cell_arrays

    assert np.allclose(grid[key], np.arange(grid.n_cells))


def test_cell_arrays_bad_value():
    with pytest.raises(TypeError):
        grid.cell_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.cell_arrays['new_array'] = np.arange(grid.n_cells - 1)


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


def test_points_np_bool():
    bool_arr = np.zeros(grid.n_points, np.bool)
    grid.point_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.point_arrays['bool_arr'].all()
    assert grid._point_scalar('bool_arr').all()
    assert grid._point_scalar('bool_arr').dtype == np.bool


def test_cells_np_bool():
    bool_arr = np.zeros(grid.n_cells, np.bool)
    grid.cell_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.cell_arrays['bool_arr'].all()
    assert grid._cell_scalar('bool_arr').all()
    assert grid._cell_scalar('bool_arr').dtype == np.bool


def test_cells_uint8():
    arr = np.zeros(grid.n_cells, np.uint8)
    grid.cell_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_cells)
    assert np.allclose(grid.cell_arrays['arr'], np.arange(grid.n_cells))


def test_points_uint8():
    arr = np.zeros(grid.n_points, np.uint8)
    grid.point_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_points)
    assert np.allclose(grid.point_arrays['arr'], np.arange(grid.n_points))


def test_bitarray_points():
    n = grid.n_points
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i%2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetPointData().AddArray(vtk_array)
    assert np.allclose(grid.point_arrays['bint_arr'], np_array)


def test_bitarray_cells():
    n = grid.n_cells
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i%2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetCellData().AddArray(vtk_array)
    assert np.allclose(grid.cell_arrays['bint_arr'], np_array)


def test_html_repr():
    """
    This just tests to make sure no errors are thrown on the HTML
    representation method for Common datasets.
    """
    repr_html = grid._repr_html_()
    assert repr_html is not None

def test_print_repr():
    """
    This just tests to make sure no errors are thrown on the text friendly
    representation method for Common datasets.
    """
    repr = grid.head()
    assert repr is not None


def test_texture():
    """Test adding texture coordinates"""
    # create a rectangle vertices
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 0.5, 0],
                         [0, 0.5, 0],])

    # mesh faces
    faces = np.hstack([[3, 0, 1, 2],
                       [3, 0, 3, 2]]).astype(np.int8)

    # Create simple texture coordinates
    t_coords = np.array([[0, 0],
                        [1, 0],
                        [1, 1],
                        [0, 1]])
    # Create the poly data
    mesh = vtki.PolyData(vertices, faces)
    # Attempt setting the texture coordinates
    mesh.t_coords = t_coords
    # now grab the texture coordinates
    foo = mesh.t_coords
    assert np.allclose(foo, t_coords)


def test_invalid_vector():
    with pytest.raises(AssertionError):
        grid.vectors = np.empty(10)

    with pytest.raises(RuntimeError):
        grid.vectors = np.empty((3, 2))

    with pytest.raises(RuntimeError):
        grid.vectors = np.empty((3, 3))


def test_no_t_coords():
    assert grid.t_coords is None


def test_no_arrows():
    assert grid.arrows is None


def test_arrows():
    sphere = vtki.Sphere(radius=3.14)

    # make cool swirly pattern
    vectors = np.vstack((np.sin(sphere.points[:, 0]),
                         np.cos(sphere.points[:, 1]),
                         np.cos(sphere.points[:, 2]))).T

    # add and scale
    assert sphere.active_vectors is None
    sphere.vectors = vectors*0.3
    assert np.allclose(sphere.active_vectors, vectors*0.3)
    assert np.allclose(sphere.vectors, vectors*0.3)

    assert sphere.active_vectors_info[1] == '_vectors'
    arrows = sphere.arrows
    assert isinstance(arrows, vtki.PolyData)
    assert np.any(arrows.points)
    sphere.set_active_vectors('_vectors')
    sphere.active_vectors_name == '_vectors'


def test_set_active_vectors_name():
    with pytest.raises(RuntimeError):
        grid.active_vectors_name = None


def test_set_active_scalars_name():
    grid.active_scalars_name = None


def test_set_t_coords():
    with pytest.raises(TypeError):
        grid.t_coords = [1, 2, 3]

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty(10)

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty((3, 3))

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty((grid.n_points, 1))

    with pytest.raises(AssertionError):
        arr = np.empty((grid.n_points, 2))
        arr[:] = -1
        grid.t_coords = arr


def test_activate_texture_none():
    assert grid._activate_texture('not a key') is None
    assert grid._activate_texture(True) is None


def test_set_active_vectors_fail():
    with pytest.raises(RuntimeError):
        grid.set_active_vectors('not a vector')


def test_set_active_scalars():
    grid_copy = grid.copy()
    arr = np.arange(grid_copy.n_cells)
    grid_copy.cell_arrays['tmp'] = arr
    grid_copy.set_active_scalar('tmp')
    assert np.allclose(grid_copy.active_scalar, arr)
    with pytest.raises(RuntimeError):
        grid_copy.set_active_scalar(None)

def test_set_active_scalar_name():
    point_keys = list(grid.point_arrays.keys())
    grid.set_active_scalar_name = point_keys[0]


def test_rename_scalar_point():
    point_keys = list(grid.point_arrays.keys())
    old_name = point_keys[0]
    new_name = 'point changed'
    grid.set_active_scalar(old_name, preference='point')
    grid.rename_scalar(old_name, new_name, preference='point')
    assert new_name in grid.point_arrays
    assert old_name not in grid.point_arrays


def test_rename_scalar_cell():
    cell_keys = list(grid.cell_arrays.keys())
    old_name = cell_keys[0]
    new_name = 'cell changed'
    grid.rename_scalar(old_name, new_name)
    assert new_name in grid.cell_arrays
    assert old_name not in grid.cell_arrays


def test_change_name_fail():
    with pytest.raises(RuntimeError):
        grid.rename_scalar('not a key', '')


def test_get_cell_scalar_fail():
    sphere = vtki.Sphere()
    with pytest.raises(RuntimeError):
        sphere._cell_scalar(name=None)


def test_extent():
    assert grid.extent is None



def set_cell_vectors():
    grid.cell_arrays['_cell_vectors'] = np.random.random((grid.n_cells, 3))
    grid.set_active_vectors('_cell_vectors')


def test_axis_rotation_invalid():
    with pytest.raises(Exception):
        vtki.common.axis_rotation(np.empty((3, 3)), 0, False, axis='not')


def test_axis_rotation_not_in_place():
    p = np.eye(3)
    p_out = vtki.common.axis_rotation(p, 1, False, axis='x')
    assert not np.allclose(p, p_out)


def test_bad_instantiation():
    with pytest.raises(TypeError):
        vtki.Common()
    with pytest.raises(TypeError):
        vtki.Grid()
    with pytest.raises(TypeError):
        vtki.DataSetFilters()
    with pytest.raises(TypeError):
        vtki.PointGrid()
    with pytest.raises(TypeError):
        vtki.ipy_tools.InteractiveTool()
    with pytest.raises(TypeError):
        vtki.BasePlotter()
