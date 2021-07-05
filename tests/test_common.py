import pickle
import numpy as np
import pytest
import vtk
from hypothesis import assume, given, settings, HealthCheck
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import composite, integers, floats, one_of
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import examples, Texture

HYPOTHESIS_MAX_EXAMPLES = 20

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


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rotate_amounts=n_numbers(4), translate_amounts=n_numbers(3))
def test_translate_should_match_vtk_transformation(rotate_amounts, translate_amounts, grid):
    trans = vtk.vtkTransform()
    trans.RotateWXYZ(*rotate_amounts)
    trans.Translate(translate_amounts)
    trans.Update()

    grid_a = grid.copy()
    grid_b = grid.copy()
    grid_c = grid.copy()
    grid_a.transform(trans)
    grid_b.transform(trans.GetMatrix())
    grid_c.transform(pyvista.array_from_vtkmatrix(trans.GetMatrix()))

    # treat INF as NAN (necessary for allclose)
    grid_a.points[np.isinf(grid_a.points)] = np.nan
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)
    assert np.allclose(grid_a.points, grid_c.points, equal_nan=True)

    # test non homogeneous transform
    trans_rotate_only = vtk.vtkTransform()
    trans_rotate_only.RotateWXYZ(*rotate_amounts)
    trans_rotate_only.Update()

    grid_d = grid.copy()
    grid_d.transform(trans_rotate_only)

    from pyvista.utilities.transformations import apply_transformation_to_points
    trans_arr = pyvista.array_from_vtkmatrix(trans_rotate_only.GetMatrix())[:3, :3]
    trans_pts = apply_transformation_to_points(trans_arr, grid.points)
    assert np.allclose(grid_d.points, trans_pts, equal_nan=True)


def test_translate_should_fail_given_none(grid):
    with pytest.raises(TypeError):
        grid.transform(None)


def test_translate_should_fail_bad_points_or_transform(grid):
    points = np.random.random((10, 2))
    bad_points = np.random.random((10, 2))
    trans = np.random.random((4, 4))
    bad_trans = np.random.random((2, 4))
    with pytest.raises(ValueError):
        pyvista.utilities.transformations.apply_transformation_to_points(trans, bad_points)

    with pytest.raises(ValueError):
        pyvista.utilities.transformations.apply_transformation_to_points(bad_trans, points)



@settings(suppress_health_check=[HealthCheck.function_scoped_fixture],
          max_examples=HYPOTHESIS_MAX_EXAMPLES)
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


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture],
          max_examples=HYPOTHESIS_MAX_EXAMPLES)
@given(angle=one_of(floats(allow_infinity=False, allow_nan=False), integers()))
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
def test_rotate_should_match_vtk_rotation(angle, axis, grid):
    trans = vtk.vtkTransform()
    getattr(trans, f'Rotate{axis.upper()}')(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pyvista.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    getattr(grid_b, f'rotate_{axis}')(angle)
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
    representation method for DataSet.
    """
    assert grid._repr_html_() is not None


@pytest.mark.parametrize('html', (True, False))
@pytest.mark.parametrize('display', (True, False))
def test_print_repr(grid, display, html):
    """
    This just tests to make sure no errors are thrown on the text friendly
    representation method for DataSet.
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
    texture = Texture(examples.mapfile)
    mesh.textures['map'] = texture
    assert mesh.textures['map'] is not None
    mesh.clear_textures()
    assert len(mesh.textures) == 0


def test_texture_airplane():
    mesh = examples.load_airplane()
    mesh.texture_map_to_plane(inplace=True, name="tex_a", use_bounds=False)
    mesh.texture_map_to_plane(inplace=True, name="tex_b", use_bounds=True)
    assert not np.allclose(mesh["tex_a"], mesh["tex_b"])
    texture = Texture(examples.mapfile)
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
        grid["vectors"] = np.empty(10)

    with pytest.raises(ValueError):
        grid["vectors"] = np.empty((3, 2))

    with pytest.raises(ValueError):
        grid["vectors"] = np.empty((3, 3))


def test_no_t_coords(grid):
    assert grid.t_coords is None


def test_no_arrows(grid):
    assert grid.arrows is None


def test_arrows():
    sphere = pyvista.Sphere(radius=3.14)

    # make cool swirly pattern
    vectors = np.vstack((np.sin(sphere.points[:, 0]),
                         np.cos(sphere.points[:, 1]),
                         np.cos(sphere.points[:, 2]))).T

    # add and scales
    sphere["vectors"] = vectors*0.3
    sphere.set_active_vectors("vectors")
    assert np.allclose(sphere.active_vectors, vectors*0.3)
    assert np.allclose(sphere["vectors"], vectors*0.3)

    assert sphere.active_vectors_info[1] == 'vectors'
    arrows = sphere.arrows
    assert isinstance(arrows, pyvista.PolyData)
    assert np.any(arrows.points)
    assert arrows.active_vectors_name == 'vectors'


def active_component_consistency_check(grid, component_type, field_association="point"):
    """
    Tests if the active component (scalars, vectors, tensors) actually reflects the underlying VTK dataset
    """
    component_type = component_type.lower()
    vtk_component_type = component_type.capitalize()

    field_association = field_association.lower()
    vtk_field_association = field_association.capitalize()

    pv_arr = getattr(grid, "active_" + component_type)
    vtk_arr = getattr(getattr(grid, "Get" + vtk_field_association + "Data")(), "Get" + vtk_component_type)()

    assert (pv_arr is None and vtk_arr is None) or np.allclose(pv_arr, vtk_to_numpy(vtk_arr))


def test_set_active_vectors(grid):
    vector_arr = np.arange(grid.n_points*3).reshape([grid.n_points, 3])
    grid.point_arrays['vector_arr'] = vector_arr
    grid.active_vectors_name = 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")
    assert grid.active_vectors_name == 'vector_arr'  
    assert np.allclose(grid.active_vectors, vector_arr)
    
    grid.active_vectors_name = None
    assert grid.active_vectors_name is None
    active_component_consistency_check(grid, "vectors", "point")


def test_set_active_tensors(grid):
    tensor_arr = np.arange(grid.n_points*9).reshape([grid.n_points, 9])
    grid.point_arrays['tensor_arr'] = tensor_arr
    grid.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")
    assert grid.active_tensors_name == 'tensor_arr'
    assert np.allclose(grid.active_tensors, tensor_arr)

    grid.active_tensors_name = None
    assert grid.active_tensors_name is None
    active_component_consistency_check(grid, "tensors", "point")


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

    active_component_consistency_check(grid, "vectors", "point")
    vector_arr = np.arange(grid.n_points * 3).reshape([grid.n_points, 3])
    grid.point_arrays['vector_arr'] = vector_arr
    grid.active_vectors_name = 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")

    grid.point_arrays['scalar_arr'] = np.zeros([grid.n_points])

    with pytest.raises(ValueError):
        grid.set_active_vectors('scalar_arr')

    assert grid.active_vectors_name == 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")


def test_set_active_tensors_fail(grid):
    with pytest.raises(ValueError):
        grid.set_active_tensors('not a tensor')

    active_component_consistency_check(grid, "tensors", "point")
    tensor_arr = np.arange(grid.n_points * 9).reshape([grid.n_points, 9])
    grid.point_arrays['tensor_arr'] = tensor_arr
    grid.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")

    grid.point_arrays['scalar_arr'] = np.zeros([grid.n_points])
    grid.point_arrays['vector_arr'] = np.zeros([grid.n_points, 3])

    with pytest.raises(ValueError):
        grid.set_active_tensors('scalar_arr')

    with pytest.raises(ValueError):
        grid.set_active_tensors('vector_arr')

    assert grid.active_tensors_name == 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")


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
        pyvista.core.dataset.axis_rotation(np.empty((3, 3)), 0, False, axis='not')


def test_axis_rotation_not_inplace():
    p = np.eye(3)
    p_out = pyvista.core.dataset.axis_rotation(p, 1, False, axis='x')
    assert not np.allclose(p, p_out)


def test_bad_instantiation():
    with pytest.raises(TypeError):
        pyvista.DataSet()
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
    arr = np.array([f'foo{i}' for i in range(10)])
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


def test_find_closest_cell():
    mesh = pyvista.Wavelet()
    node = np.array([0, 0.2, 0.2])

    with pytest.raises(ValueError):
        mesh.find_closest_cell([1, 2])

    with pytest.raises(TypeError):
        # allow Sequence but not Iterable
        mesh.find_closest_cell({1, 2, 3})

    # array but bad size
    with pytest.raises(ValueError):
        mesh.find_closest_cell(np.empty(4))

    index = mesh.find_closest_cell(node)
    assert isinstance(index, int)


def test_find_closest_cells():
    mesh = pyvista.Sphere()
    # invalid array dim
    with pytest.raises(ValueError):
        mesh.find_closest_cell(np.empty((1, 1, 1)))

    # test invalid array size
    with pytest.raises(ValueError):
        mesh.find_closest_cell(np.empty((4, 4)))

    # simply get the face centers
    fcent = mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]].mean(1)
    indices = mesh.find_closest_cell(fcent)

    # this will miss a few...
    mask = indices == -1
    assert mask.sum() < 10

    # Make sure we match the face centers
    assert np.allclose(indices[~mask], np.arange(mesh.n_faces)[~mask])


def test_setting_points_from_self(grid):
    grid_copy = grid.copy()
    grid.points = grid_copy.points
    assert np.allclose(grid.points, grid_copy.points)


def test_empty_points():
    pdata = pyvista.PolyData()
    assert np.allclose(pdata.points, np.empty(3))


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
    rng = grid.get_data_range()  # active scalars
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = grid.get_data_range('sample_point_scalars')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = grid.get_data_range('sample_cell_scalars')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 40))


def test_actual_memory_size(grid):
    size = grid.actual_memory_size
    assert isinstance(size, int)
    assert size >= 0


def test_copy_structure(grid):
    classname = grid.__class__.__name__
    copy = eval(f'pyvista.{classname}')()
    copy.copy_structure(grid)
    assert copy.n_cells == grid.n_cells
    assert copy.n_points == grid.n_points
    assert len(copy.field_arrays) == 0
    assert len(copy.cell_arrays) == 0
    assert len(copy.point_arrays) == 0


def test_copy_attributes(grid):
    classname = grid.__class__.__name__
    copy = eval(f'pyvista.{classname}')()
    copy.copy_attributes(grid)
    assert copy.n_cells == 0
    assert copy.n_points == 0
    assert copy.field_arrays.keys() == grid.field_arrays.keys()
    assert copy.cell_arrays.keys() == grid.cell_arrays.keys()
    assert copy.point_arrays.keys() == grid.point_arrays.keys()


def test_cell_n_points(grid):
    npoints = grid.cell_n_points(0)
    assert isinstance(npoints, int)
    assert npoints >= 0


def test_cell_points(grid):
    points = grid.cell_points(0)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3


def test_cell_bounds(grid):
    bounds = grid.cell_bounds(0)
    assert isinstance(bounds, list)
    assert len(bounds) == 6


def test_cell_type(grid):
    ctype = grid.cell_type(0)
    assert isinstance(ctype, int)



def test_serialize_deserialize(datasets):
    for dataset in datasets:
        dataset_2 = pickle.loads(pickle.dumps(dataset))

        # check python attributes are the same
        for attr in dataset.__dict__:
            assert getattr(dataset_2, attr) == getattr(dataset, attr)

        # check data is the same
        for attr in ('n_cells', 'n_points', 'n_arrays'):
            if hasattr(dataset, attr):
                assert getattr(dataset_2, attr) == getattr(dataset, attr)

        for attr in ('cells', 'points'):
            if hasattr(dataset, attr):
                arr_have = getattr(dataset_2, attr)
                arr_expected = getattr(dataset, attr)
                assert arr_have == pytest.approx(arr_expected)

        for name in dataset.point_arrays:
            arr_have = dataset_2.point_arrays[name]
            arr_expected = dataset.point_arrays[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.cell_arrays:
            arr_have = dataset_2.cell_arrays[name]
            arr_expected = dataset.cell_arrays[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.field_arrays:
            arr_have = dataset_2.field_arrays[name]
            arr_expected = dataset.field_arrays[name]
            assert arr_have == pytest.approx(arr_expected)


def test_rotations_should_match_by_a_360_degree_difference():
    mesh = examples.load_airplane()

    point = np.random.random(3) - 0.5
    angle = (np.random.random() - 0.5) * 360.0
    vector = np.random.random(3) - 0.5

    # Rotate about x axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_x(angle=angle, point=point)
    rot2.rotate_x(angle=angle - 360.0, point=point)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about y axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_y(angle=angle, point=point)
    rot2.rotate_y(angle=angle - 360.0, point=point)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about z axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_z(angle=angle, point=point)
    rot2.rotate_z(angle=angle - 360.0, point=point)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about custom vector.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_vector(vector=vector, angle=angle, point=point)
    rot2.rotate_vector(vector=vector, angle=angle - 360.0, point=point)
    assert np.allclose(rot1.points, rot2.points)
