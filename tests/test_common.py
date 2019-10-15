import sys

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import examples

GRID = pyvista.UnstructuredGrid(examples.hexbeamfile)

py2 = sys.version_info.major == 2


def test_point_arrays():
    key = 'test_array_points'
    grid = GRID.copy()
    grid[key] = np.arange(grid.n_points)
    assert key in grid.point_arrays

    orig_value = grid.point_arrays[key][0]/1.0
    grid.point_arrays[key][0] += 1
    assert orig_value == grid._point_array(key)[0] - 1

    del grid.point_arrays[key]
    assert key not in grid.point_arrays

    grid.point_arrays[key] = np.arange(grid.n_points)
    assert key in grid.point_arrays

    assert np.allclose(grid[key], np.arange(grid.n_points))

    grid.clear_point_arrays()
    assert len(grid.point_arrays.keys()) == 0

    grid.point_arrays['list'] = np.arange(grid.n_points).tolist()
    assert isinstance(grid.point_arrays['list'], np.ndarray)
    assert np.allclose(grid.point_arrays['list'], np.arange(grid.n_points))


def test_point_arrays_bad_value():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.point_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.point_arrays['new_array'] = np.arange(grid.n_points - 1)


def test_cell_arrays():
    key = 'test_array_cells'
    grid = GRID.copy()
    grid[key] = np.arange(grid.n_cells)
    assert key in grid.cell_arrays

    orig_value = grid.cell_arrays[key][0]/1.0
    grid.cell_arrays[key][0] += 1
    assert orig_value == grid.cell_arrays[key][0] - 1

    del grid.cell_arrays[key]
    assert key not in grid.cell_arrays

    grid.cell_arrays[key] = np.arange(grid.n_cells)
    assert key in grid.cell_arrays

    assert np.allclose(grid[key], np.arange(grid.n_cells))

    grid.cell_arrays['list'] = np.arange(grid.n_cells).tolist()
    assert isinstance(grid.cell_arrays['list'], np.ndarray)
    assert np.allclose(grid.cell_arrays['list'], np.arange(grid.n_cells))


def test_cell_arrays_bad_value():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.cell_arrays['new_array'] = None

    with pytest.raises(Exception):
        grid.cell_arrays['new_array'] = np.arange(grid.n_cells - 1)


def test_field_arrays():
    key = 'test_array_field'
    grid = GRID.copy()
    # Add array of length not equal to n_cells or n_points
    n = grid.n_cells // 3
    grid.field_arrays[key] = np.arange(n)
    assert key in grid.field_arrays
    assert np.allclose(grid.field_arrays[key], np.arange(n))
    assert np.allclose(grid[key], np.arange(n))

    orig_value = grid.field_arrays[key][0]/1.0
    grid.field_arrays[key][0] += 1
    assert orig_value == grid.field_arrays[key][0] - 1

    assert key in grid.array_names

    del grid.field_arrays[key]
    assert key not in grid.field_arrays

    grid.field_arrays['list'] = np.arange(n).tolist()
    assert isinstance(grid.field_arrays['list'], np.ndarray)
    assert np.allclose(grid.field_arrays['list'], np.arange(n))




def test_field_arrays_bad_value():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.field_arrays['new_array'] = None


def test_copy():
    grid = GRID.copy()
    grid_copy = grid.copy(deep=True)
    grid_copy.points[0] = np.nan
    assert not np.any(np.isnan(grid.points[0]))

    grid_copy_shallow = grid.copy(deep=False)
    grid_copy.points[0] += 0.1
    assert np.all(grid_copy_shallow.points[0] == grid.points[0])


def test_transform():
    grid = GRID.copy()
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
    grid_c.transform(pyvista.trans_from_matrix(trans.GetMatrix()))
    assert np.allclose(grid_a.points, grid_b.points)
    assert np.allclose(grid_a.points, grid_c.points)


def test_transform_errors():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.transform(None)

    with pytest.raises(Exception):
        grid.transform(np.array([1]))


def test_translate():
    grid = GRID.copy()
    grid_copy = grid.copy()
    xyz = [1, 1, 1]
    grid_copy.translate(xyz)

    grid_points = grid.points.copy() + np.array(xyz)
    assert np.allclose(grid_copy.points, grid_points)


def test_rotate_x():
    grid = GRID.copy()
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateX(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pyvista.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_x(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_rotate_y():
    grid = GRID.copy()
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateY(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pyvista.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_y(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_rotate_z():
    grid = GRID.copy()
    angle = 30
    trans = vtk.vtkTransform()
    trans.RotateZ(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pyvista.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    grid_b.rotate_z(angle)
    assert np.allclose(grid_a.points, grid_b.points)


def test_make_points_double():
    grid = GRID.copy()
    grid_copy = grid.copy()
    grid_copy.points = grid_copy.points.astype(np.float32)
    assert grid_copy.points.dtype == np.float32
    grid_copy.points_to_double()
    assert grid_copy.points.dtype == np.double


def test_invalid_points():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.points = None


def test_points_np_bool():
    grid = GRID.copy()
    bool_arr = np.zeros(grid.n_points, np.bool)
    grid.point_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.point_arrays['bool_arr'].all()
    assert grid._point_array('bool_arr').all()
    assert grid._point_array('bool_arr').dtype == np.bool


def test_cells_np_bool():
    grid = GRID.copy()
    bool_arr = np.zeros(grid.n_cells, np.bool)
    grid.cell_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.cell_arrays['bool_arr'].all()
    assert grid._cell_array('bool_arr').all()
    assert grid._cell_array('bool_arr').dtype == np.bool


def test_field_np_bool():
    grid = GRID.copy()
    bool_arr = np.zeros(grid.n_cells // 3, np.bool)
    grid.field_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.field_arrays['bool_arr'].all()
    assert grid._field_array('bool_arr').all()
    assert grid._field_array('bool_arr').dtype == np.bool


def test_cells_uint8():
    grid = GRID.copy()
    arr = np.zeros(grid.n_cells, np.uint8)
    grid.cell_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_cells)
    assert np.allclose(grid.cell_arrays['arr'], np.arange(grid.n_cells))


def test_points_uint8():
    grid = GRID.copy()
    arr = np.zeros(grid.n_points, np.uint8)
    grid.point_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_points)
    assert np.allclose(grid.point_arrays['arr'], np.arange(grid.n_points))


def test_field_uint8():
    grid = GRID.copy()
    n = grid.n_points//3
    arr = np.zeros(n, np.uint8)
    grid.field_arrays['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(grid.field_arrays['arr'], np.arange(n))


def test_bitarray_points():
    grid = GRID.copy()
    n = grid.n_points
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetPointData().AddArray(vtk_array)
    assert np.allclose(grid.point_arrays['bint_arr'], np_array)


def test_bitarray_cells():
    grid = GRID.copy()
    n = grid.n_cells
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetCellData().AddArray(vtk_array)
    assert np.allclose(grid.cell_arrays['bint_arr'], np_array)


def test_bitarray_field():
    grid = GRID.copy()
    n = grid.n_cells // 3
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetFieldData().AddArray(vtk_array)
    assert np.allclose(grid.field_arrays['bint_arr'], np_array)


def test_html_repr():
    """
    This just tests to make sure no errors are thrown on the HTML
    representation method for Common datasets.
    """
    grid = GRID.copy()
    repr_html = grid._repr_html_()
    assert repr_html is not None

def test_print_repr():
    """
    This just tests to make sure no errors are thrown on the text friendly
    representation method for Common datasets.
    """
    grid = GRID.copy()
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
    mesh = pyvista.PolyData(vertices, faces)
    # Attempt setting the texture coordinates
    mesh.t_coords = t_coords
    # now grab the texture coordinates
    foo = mesh.t_coords
    assert np.allclose(foo, t_coords)
    texture = pyvista.read_texture(examples.mapfile)
    mesh.textures['map'] = texture
    assert mesh.textures['map'] is not None
    mesh.clear_textures()
    assert len(mesh.textures) == 0



def test_invalid_vector():
    grid = GRID.copy()
    with pytest.raises(AssertionError):
        grid.vectors = np.empty(10)

    with pytest.raises(RuntimeError):
        grid.vectors = np.empty((3, 2))

    with pytest.raises(RuntimeError):
        grid.vectors = np.empty((3, 3))


def test_no_t_coords():
    grid = GRID.copy()
    assert grid.t_coords is None


def test_no_arrows():
    grid = GRID.copy()
    assert grid.arrows is None


def test_arrows():
    grid = GRID.copy()
    sphere = pyvista.Sphere(radius=3.14)

    # make cool swirly pattern
    vectors = np.vstack((np.sin(sphere.points[:, 0]),
                         np.cos(sphere.points[:, 1]),
                         np.cos(sphere.points[:, 2]))).T

    # add and scales
    sphere.vectors = vectors*0.3
    assert np.allclose(sphere.active_vectors, vectors*0.3)
    assert np.allclose(sphere.vectors, vectors*0.3)

    assert sphere.active_vectors_info[1] == '_vectors'
    arrows = sphere.arrows
    assert isinstance(arrows, pyvista.PolyData)
    assert np.any(arrows.points)
    sphere.set_active_vectors('_vectors')
    sphere.active_vectors_name == '_vectors'


def test_set_active_vectors_name():
    grid = GRID.copy()
    grid.active_vectors_name = None


def test_set_active_scalars_name():
    grid = GRID.copy()
    grid.active_scalars_name = None


def test_set_t_coords():
    grid = GRID.copy()
    with pytest.raises(TypeError):
        grid.t_coords = [1, 2, 3]

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty(10)

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty((3, 3))

    with pytest.raises(AssertionError):
        grid.t_coords = np.empty((grid.n_points, 1))


def test_activate_texture_none():
    grid = GRID.copy()
    assert grid._activate_texture('not a key') is None
    assert grid._activate_texture(True) is None


def test_set_active_vectors_fail():
    grid = GRID.copy()
    with pytest.raises(RuntimeError):
        grid.set_active_vectors('not a vector')


def test_set_active_scalars():
    grid = GRID.copy()
    grid_copy = grid.copy()
    arr = np.arange(grid_copy.n_cells)
    grid_copy.cell_arrays['tmp'] = arr
    grid_copy.set_active_scalar('tmp')
    assert np.allclose(grid_copy.active_scalar, arr)
    # Make sure we can set no active scalars
    grid_copy.set_active_scalar(None)
    assert grid_copy.GetPointData().GetScalars() is None
    assert grid_copy.GetCellData().GetScalars() is None

def test_set_active_scalar_name():
    grid = GRID.copy()
    point_keys = list(grid.point_arrays.keys())
    grid.set_active_scalar_name = point_keys[0]


def test_rename_scalar_point():
    grid = GRID.copy()
    point_keys = list(grid.point_arrays.keys())
    old_name = point_keys[0]
    new_name = 'point changed'
    grid.set_active_scalar(old_name, preference='point')
    grid.rename_scalar(old_name, new_name, preference='point')
    assert new_name in grid.point_arrays
    assert old_name not in grid.point_arrays


def test_rename_scalar_cell():
    grid = GRID.copy()
    cell_keys = list(grid.cell_arrays.keys())
    old_name = cell_keys[0]
    new_name = 'cell changed'
    grid.rename_scalar(old_name, new_name)
    assert new_name in grid.cell_arrays
    assert old_name not in grid.cell_arrays


def test_rename_scalar_field():
    grid = GRID.copy()
    grid.field_arrays['fieldfoo'] = np.array([8, 6, 7])
    field_keys = list(grid.field_arrays.keys())
    old_name = field_keys[0]
    new_name = 'cell changed'
    grid.rename_scalar(old_name, new_name)
    assert new_name in grid.field_arrays
    assert old_name not in grid.field_arrays


def test_change_name_fail():
    grid = GRID.copy()
    with pytest.raises(RuntimeError):
        grid.rename_scalar('not a key', '')


def test_get_cell_array_fail():
    sphere = pyvista.Sphere()
    with pytest.raises(RuntimeError):
        sphere._cell_array(name=None)


def test_extent():
    grid = GRID.copy()
    assert grid.extent is None



def set_cell_vectors():
    grid = GRID.copy()
    grid.cell_arrays['_cell_vectors'] = np.random.random((grid.n_cells, 3))
    grid.set_active_vectors('_cell_vectors')


def test_axis_rotation_invalid():
    with pytest.raises(Exception):
        pyvista.core.common.axis_rotation(np.empty((3, 3)), 0, False, axis='not')


def test_axis_rotation_not_inplace():
    p = np.eye(3)
    p_out = pyvista.core.common.axis_rotation(p, 1, False, axis='x')
    assert not np.allclose(p, p_out)


def test_bad_instantiation():
    with pytest.raises(TypeError):
        pyvista.Common()
    with pytest.raises(TypeError):
        pyvista.Grid()
    with pytest.raises(TypeError):
        pyvista.DataSetFilters()
    with pytest.raises(TypeError):
        pyvista.PointGrid()
    with pytest.raises(TypeError):
        pyvista.BasePlotter()


def test_string_arrays():
    poly = pyvista.PolyData(np.random.rand(10, 3))
    arr = np.array(['foo{}'.format(i) for i in range(10)])
    poly['foo'] = arr
    back = poly['foo']
    assert len(back) == 10


def test_clear_arrays():
    # First try on an empy mesh
    grid = pyvista.UniformGrid((10, 10, 10))
    grid.clear_arrays()
    # Now try something more complicated
    grid = GRID.copy()
    grid.clear_arrays()
    grid['foo-p'] = np.random.rand(grid.n_points)
    grid['foo-c'] = np.random.rand(grid.n_cells)
    grid.field_arrays['foo-f'] = np.random.rand(grid.n_points * grid.n_cells)
    assert grid.n_arrays == 3
    grid.clear_arrays()
    assert grid.n_arrays == 0


def test_scalars_dict_update():
    mesh = examples.load_uniform()
    n = len(mesh.point_arrays)
    arrays = {
        'foo': np.arange(mesh.n_points),
        'rand': np.random.random(mesh.n_points)
    }
    mesh.point_arrays.update(arrays)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_arrays) == n + 2

    # Test update from Table
    table = pyvista.Table(arrays)
    mesh = examples.load_uniform()
    mesh.point_arrays.update(table)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_arrays) == n + 2


def test_hanlde_array_with_null_name():
    poly = pyvista.PolyData()
    # Add point array with no name
    poly.GetPointData().AddArray(pyvista.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    pdata = poly.point_arrays
    assert pdata is not None
    assert len(pdata) == 1
    # Add cell array with no name
    poly.GetCellData().AddArray(pyvista.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    cdata = poly.cell_arrays
    assert cdata is not None
    assert len(cdata) == 1
    # Add field array with no name
    poly.GetFieldData().AddArray(pyvista.convert_array(np.array([5, 6])))
    html = poly._repr_html_()
    assert html is not None
    fdata = poly.field_arrays
    assert fdata is not None
    assert len(fdata) == 1



def test_shallow_copy_back_propagation():
    """Test that the original data object's points get modified after a
    shallow copy.

    Reference: https://github.com/pyvista/pyvista/issues/375#issuecomment-531691483
    """
    # Case 1
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(2.0, 0.0, 0.0)
    original = vtk.vtkPolyData()
    original.SetPoints(points)
    wrapped = pyvista.PolyData(original, deep=False)
    wrapped.points[:] = 2.8
    orig_points = vtk_to_numpy(original.GetPoints().GetData())
    assert np.allclose(orig_points, wrapped.points)
    # Case 2
    original = vtk.vtkPolyData()
    wrapped = pyvista.PolyData(original, deep=False)
    wrapped.points = np.random.rand(5, 3)
    orig_points = vtk_to_numpy(original.GetPoints().GetData())
    assert np.allclose(orig_points, wrapped.points)
