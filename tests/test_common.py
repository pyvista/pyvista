import numpy as np
import pytest
import vtk
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import composite, integers, floats, one_of
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import examples


@pytest.fixture()
def grid():
    return pyvista.UnstructuredGrid(examples.hexbeamfile)


def test_invalid_overwrite(grid):
    with pytest.raises(TypeError):
        grid.overwrite(pyvista.Plane())


@composite
def n_numbers(draw, n):
    numbers = []
    for _ in range(n):
        number = draw(one_of(floats(), integers()))
        numbers.append(number)
    return numbers


def test_memory_address(grid):
    assert isinstance(grid.memory_address, str)
    assert 'Addr' in grid.memory_address


def test_point_arrays(grid):
    key = 'test_array_points'
    grid[key] = np.arange(grid.n_points)
    assert key in grid.point_arrays

    orig_value = grid.point_arrays[key][0]/1.0
    grid.point_arrays[key][0] += 1
    assert orig_value == grid.point_arrays[key][0] - 1

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


def test_point_arrays_bad_value(grid):
    with pytest.raises(TypeError):
        grid.point_arrays['new_array'] = None

    with pytest.raises(ValueError):
        grid.point_arrays['new_array'] = np.arange(grid.n_points - 1)


def test_ipython_key_completions(grid):
    assert isinstance(grid._ipython_key_completions_(), list)


def test_cell_arrays(grid):
    key = 'test_array_cells'
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


def test_cell_array_range(grid):
    rng = range(grid.n_cells)
    grid.cell_arrays['tmp'] = rng
    assert np.allclose(rng, grid.cell_arrays['tmp'])


def test_cell_arrays_bad_value(grid):
    with pytest.raises(TypeError):
        grid.cell_arrays['new_array'] = None

    with pytest.raises(ValueError):
        grid.cell_arrays['new_array'] = np.arange(grid.n_cells - 1)


def test_field_arrays(grid):
    key = 'test_array_field'
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

    foo = np.arange(n) * 5
    grid.add_field_array(foo, 'foo')
    assert isinstance(grid.field_arrays['foo'], np.ndarray)
    assert np.allclose(grid.field_arrays['foo'], foo)

    with pytest.raises(ValueError):
        grid.set_active_scalars('foo')


@pytest.mark.parametrize('field', (range(5), np.ones((3,3))[:, 0]))
def test_add_field_array(grid, field):
    grid.add_field_array(field, 'foo')
    assert isinstance(grid.field_arrays['foo'], np.ndarray)
    assert np.allclose(grid.field_arrays['foo'], field)


def test_modify_field_array(grid):
    field = range(4)
    grid.add_field_array(range(5), 'foo')
    grid.add_field_array(field, 'foo')
    assert np.allclose(grid.field_arrays['foo'], field)

    field = range(8)
    grid.field_arrays['foo'] = field
    assert np.allclose(grid.field_arrays['foo'], field)


def test_active_scalars_cell(grid):
    grid.add_field_array(range(5), 'foo')
    del grid.point_arrays['sample_point_scalars']
    del grid.point_arrays['VTKorigID']
    assert grid.active_scalars_info[1] == 'sample_cell_scalars'


def test_field_arrays_bad_value(grid):
    with pytest.raises(TypeError):
        grid.field_arrays['new_array'] = None


def test_copy(grid):
    grid_copy = grid.copy(deep=True)
    grid_copy.points[0] = np.nan
    assert not np.any(np.isnan(grid.points[0]))

    grid_copy_shallow = grid.copy(deep=False)
    grid_copy.points[0] += 0.1
    assert np.all(grid_copy_shallow.points[0] == grid.points[0])


@given(rotate_amounts=n_numbers(3), translate_amounts=n_numbers(3))
def test_translate_should_match_vtk_transformation(rotate_amounts, translate_amounts, grid):
    trans = vtk.vtkTransform()
    trans.RotateWXYZ(0, *rotate_amounts)
    trans.Translate(translate_amounts)
    trans.Update()

    grid_a = grid.copy()
    grid_b = grid.copy()
    grid_c = grid.copy()
    grid_a.transform(trans)
    grid_b.transform(trans.GetMatrix())
    grid_c.transform(pyvista.trans_from_matrix(trans.GetMatrix()))
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)
    assert np.allclose(grid_a.points, grid_c.points, equal_nan=True)


def test_translate_should_fail_given_none(grid):
    with pytest.raises(TypeError):
        grid.transform(None)


@given(array=arrays(dtype=np.float32, shape=array_shapes(max_dims=5, max_side=5)))
def test_transform_should_fail_given_wrong_numpy_shape(array, grid):
    assume(array.shape != (4, 4))
    with pytest.raises(ValueError):
        grid.transform(array)


@pytest.mark.parametrize('axis_amounts', [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
def test_translate_should_translate_grid(grid, axis_amounts):
    grid_copy = grid.copy()
    grid_copy.translate(axis_amounts)

    grid_points = grid.points.copy() + np.array(axis_amounts)
    assert np.allclose(grid_copy.points, grid_points)


@given(angle=one_of(floats(allow_infinity=False, allow_nan=False), integers()))
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_rotate_should_match_vtk_rotation(angle, axis, grid):
    trans = vtk.vtkTransform()
    getattr(trans, 'Rotate{}'.format(axis.upper()))(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pyvista.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    getattr(grid_b, 'rotate_{}'.format(axis))(angle)
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)


def test_make_points_double(grid):
    grid.points = grid.points.astype(np.float32)
    assert grid.points.dtype == np.float32
    grid.points_to_double()
    assert grid.points.dtype == np.double


def test_invalid_points(grid):
    with pytest.raises(TypeError):
        grid.points = None


def test_points_np_bool(grid):
    bool_arr = np.zeros(grid.n_points, np.bool_)
    grid.point_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.point_arrays['bool_arr'].all()
    assert grid.point_arrays['bool_arr'].all()
    assert grid.point_arrays['bool_arr'].dtype == np.bool_


def test_cells_np_bool(grid):
    bool_arr = np.zeros(grid.n_cells, np.bool_)
    grid.cell_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.cell_arrays['bool_arr'].all()
    assert grid.cell_arrays['bool_arr'].all()
    assert grid.cell_arrays['bool_arr'].dtype == np.bool_


def test_field_np_bool(grid):
    bool_arr = np.zeros(grid.n_cells // 3, np.bool_)
    grid.field_arrays['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.field_arrays['bool_arr'].all()
    assert grid.field_arrays['bool_arr'].all()
    assert grid.field_arrays['bool_arr'].dtype == np.bool_


def test_cells_uint8(grid):
    arr = np.zeros(grid.n_cells, np.uint8)
    grid.cell_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_cells)
    assert np.allclose(grid.cell_arrays['arr'], np.arange(grid.n_cells))


def test_points_uint8(grid):
    arr = np.zeros(grid.n_points, np.uint8)
    grid.point_arrays['arr'] = arr
    arr[:] = np.arange(grid.n_points)
    assert np.allclose(grid.point_arrays['arr'], np.arange(grid.n_points))


def test_field_uint8(grid):
    n = grid.n_points//3
    arr = np.zeros(n, np.uint8)
    grid.field_arrays['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(grid.field_arrays['arr'], np.arange(n))


def test_bitarray_points(grid):
    n = grid.n_points
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetPointData().AddArray(vtk_array)
    assert np.allclose(grid.point_arrays['bint_arr'], np_array)


def test_bitarray_cells(grid):
    n = grid.n_cells
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetCellData().AddArray(vtk_array)
    assert np.allclose(grid.cell_arrays['bint_arr'], np_array)


def test_bitarray_field(grid):
    n = grid.n_cells // 3
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    grid.GetFieldData().AddArray(vtk_array)
    assert np.allclose(grid.field_arrays['bint_arr'], np_array)


def test_html_repr(grid):
    """
    This just tests to make sure no errors are thrown on the HTML
    representation method for Common datasets.
    """
    assert grid._repr_html_() is not None


@pytest.mark.parametrize('html', (True, False))
@pytest.mark.parametrize('display', (True, False))
def test_print_repr(grid, display, html):
    """
    This just tests to make sure no errors are thrown on the text friendly
    representation method for Common datasets.
    """
    result = grid.head(display=display, html=html)
    if display and html:
        assert result is None
    else:
        assert result is not None


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


def test_texture_airplane():
    mesh = examples.load_airplane()
    mesh.texture_map_to_plane(inplace=True, name="tex_a", use_bounds=False)
    mesh.texture_map_to_plane(inplace=True, name="tex_b", use_bounds=True)
    assert not np.allclose(mesh["tex_a"], mesh["tex_b"])
    texture = pyvista.read_texture(examples.mapfile)
    mesh.textures["tex_a"] = texture.copy()
    mesh.textures["tex_b"] = texture.copy()
    mesh._activate_texture("tex_a")
    assert np.allclose(mesh.t_coords, mesh["tex_a"])
    mesh._activate_texture("tex_b")
    assert np.allclose(mesh.t_coords, mesh["tex_b"])

    # Now test copying
    cmesh = mesh.copy()
    assert len(cmesh.textures) == 2
    assert "tex_a" in cmesh.textures
    assert "tex_b" in cmesh.textures


def test_invalid_vector(grid):
    with pytest.raises(ValueError):
        grid.vectors = np.empty(10)

    with pytest.raises(ValueError):
        grid.vectors = np.empty((3, 2))

    with pytest.raises(ValueError):
        grid.vectors = np.empty((3, 3))


def test_no_t_coords(grid):
    assert grid.t_coords is None


def test_no_arrows(grid):
    assert grid.arrows is None


def test_arrows(grid):
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
    assert sphere.active_vectors_name == '_vectors'


def test_set_active_vectors_name(grid):
    grid.active_vectors_name = None
    assert grid.active_vectors_name is None


def test_set_t_coords(grid):
    with pytest.raises(TypeError):
        grid.t_coords = [1, 2, 3]

    with pytest.raises(ValueError):
        grid.t_coords = np.empty(10)

    with pytest.raises(ValueError):
        grid.t_coords = np.empty((3, 3))

    with pytest.raises(ValueError):
        grid.t_coords = np.empty((grid.n_points, 1))


def test_activate_texture_none(grid):
    assert grid._activate_texture('not a key') is None
    assert grid._activate_texture(True) is None


def test_set_active_vectors_fail(grid):
    with pytest.raises(ValueError):
        grid.set_active_vectors('not a vector')


def test_set_active_scalars(grid):
    arr = np.arange(grid.n_cells)
    grid.cell_arrays['tmp'] = arr
    grid.set_active_scalars('tmp')
    assert np.allclose(grid.active_scalars, arr)
    # Make sure we can set no active scalars
    grid.set_active_scalars(None)
    assert grid.GetPointData().GetScalars() is None
    assert grid.GetCellData().GetScalars() is None


def test_set_active_scalars_name(grid):
    point_keys = list(grid.point_arrays.keys())
    grid.active_scalars_name = point_keys[0]
    grid.active_scalars_name = None


def test_rename_array_point(grid):
    point_keys = list(grid.point_arrays.keys())
    old_name = point_keys[0]
    new_name = 'point changed'
    grid.set_active_scalars(old_name, preference='point')
    grid.rename_array(old_name, new_name, preference='point')
    assert new_name in grid.point_arrays
    assert old_name not in grid.point_arrays
    assert new_name == grid.active_scalars_name


def test_rename_array_cell(grid):
    cell_keys = list(grid.cell_arrays.keys())
    old_name = cell_keys[0]
    new_name = 'cell changed'
    grid.rename_array(old_name, new_name)
    assert new_name in grid.cell_arrays
    assert old_name not in grid.cell_arrays


def test_rename_array_field(grid):
    grid.field_arrays['fieldfoo'] = np.array([8, 6, 7])
    field_keys = list(grid.field_arrays.keys())
    old_name = field_keys[0]
    new_name = 'cell changed'
    grid.rename_array(old_name, new_name)
    assert new_name in grid.field_arrays
    assert old_name not in grid.field_arrays


def test_change_name_fail(grid):
    with pytest.raises(KeyError):
        grid.rename_array('not a key', '')


def test_get_cell_array_fail():
    sphere = pyvista.Sphere()
    with pytest.raises(KeyError):
        sphere.cell_arrays[None]


def test_extent_none(grid):
    assert grid.extent is None


def test_set_extent_expect_error(grid):
    with pytest.raises(AttributeError):
        grid.extent = [1, 2, 3]


def test_set_extent():
    dims = [10, 10, 10]
    uni_grid = pyvista.UniformGrid(dims)
    with pytest.raises(ValueError):
        uni_grid.extent = [0, 1]

    extent = [0, 1, 0, 1, 0, 1]
    uni_grid.extent = extent
    assert np.allclose(uni_grid.extent, extent)


def test_get_item(grid):
    with pytest.raises(KeyError):
        grid[0]


def test_set_item(grid):
    with pytest.raises(TypeError):
        grid['tmp'] = None

    # field data
    with pytest.raises(ValueError):
        grid['bad_field'] = range(5)


def test_set_item_range(grid):
    rng = range(grid.n_points)
    grid['pt_rng'] = rng
    assert np.allclose(grid['pt_rng'], rng)


def test_str(grid):
    assert 'UnstructuredGrid' in str(grid)


def test_set_cell_vectors(grid):
    arr = np.random.random((grid.n_cells, 3))
    grid.cell_arrays['_cell_vectors'] = arr
    grid.set_active_vectors('_cell_vectors')
    assert grid.active_vectors_name == '_cell_vectors'
    assert np.allclose(grid.active_vectors, arr)


def test_axis_rotation_invalid():
    with pytest.raises(ValueError):
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
    with pytest.raises(TypeError):
        pyvista.DataObject()


def test_string_arrays():
    poly = pyvista.PolyData(np.random.rand(10, 3))
    arr = np.array(['foo{}'.format(i) for i in range(10)])
    poly['foo'] = arr
    back = poly['foo']
    assert len(back) == 10


def test_clear_arrays():
    # First try on an empty mesh
    grid = pyvista.UniformGrid((10, 10, 10))
    # Now try something more complicated
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


def test_handle_array_with_null_name():
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


def test_add_point_array_list(grid):
    rng = range(grid.n_points)
    grid.point_arrays['tmp'] = rng
    assert np.allclose(grid.point_arrays['tmp'], rng)


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


def test_find_closest_point():
    sphere = pyvista.Sphere()
    node = np.array([0, 0.2, 0.2])

    with pytest.raises(TypeError):
        sphere.find_closest_point([1, 2])

    with pytest.raises(ValueError):
        sphere.find_closest_point([0, 0, 0], n=0)

    with pytest.raises(TypeError):
        sphere.find_closest_point([0, 0, 0], n=3.0)

    with pytest.raises(TypeError):
        # allow Sequence but not Iterable
        sphere.find_closest_point({1, 2, 3})

    index = sphere.find_closest_point(node)
    assert isinstance(index, int)
    # Make sure we can fetch that point
    closest = sphere.points[index]
    assert len(closest) == 3
    # n points
    node = np.array([0, 0.2, 0.2])
    index = sphere.find_closest_point(node, 5)
    assert len(index) == 5


def test_setting_points_from_self(grid):
    grid_copy = grid.copy()
    grid.points = grid_copy.points
    assert np.allclose(grid.points, grid_copy.points)


def test_empty_points():
    pdata = pyvista.PolyData()
    assert pdata.points is None


def test_no_active():
    pdata = pyvista.PolyData()
    assert pdata.active_scalars is None

    with pytest.raises(KeyError):
        pdata.point_arrays[None]


def test_get_data_range(grid):
    # Test with blank mesh
    mesh = pyvista.Sphere()
    mesh.clear_arrays()
    rng = mesh.get_data_range()
    assert all(np.isnan(rng))
    with pytest.raises(ValueError):
        rng = mesh.get_data_range('some data')

    # Test with some data
    rng = grid.get_data_range() # active scalars
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = grid.get_data_range('sample_point_scalars')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = grid.get_data_range('sample_cell_scalars')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 40))
