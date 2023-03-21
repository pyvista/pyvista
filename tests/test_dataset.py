"""Tests for pyvista.core.dataset."""

import multiprocessing
import pickle

from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import composite, floats, integers, one_of
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import Texture, examples
from pyvista.core.dataset import DataSet
from pyvista.core.errors import VTKVersionError
from pyvista.examples import (
    load_airplane,
    load_explicit_structured,
    load_hexbeam,
    load_rectilinear,
    load_structured,
    load_tetbeam,
    load_uniform,
)
from pyvista.utilities.misc import PyVistaDeprecationWarning

HYPOTHESIS_MAX_EXAMPLES = 20


@pytest.fixture()
def grid():
    return pyvista.UnstructuredGrid(examples.hexbeamfile)


def test_invalid_overwrite(grid):
    with pytest.raises(TypeError):
        grid.copy_from(pyvista.Plane())


def test_overwrite_deprecation(grid):
    mesh = type(grid)()
    with pytest.warns(PyVistaDeprecationWarning):
        mesh.overwrite(grid)


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


def test_point_data(grid):
    key = 'test_array_points'
    grid[key] = np.arange(grid.n_points)
    assert key in grid.point_data

    orig_value = grid.point_data[key][0] / 1.0
    grid.point_data[key][0] += 1
    assert orig_value == grid.point_data[key][0] - 1

    del grid.point_data[key]
    assert key not in grid.point_data

    grid.point_data[key] = np.arange(grid.n_points)
    assert key in grid.point_data

    assert np.allclose(grid[key], np.arange(grid.n_points))

    grid.clear_point_data()
    assert len(grid.point_data.keys()) == 0

    grid.point_data['list'] = np.arange(grid.n_points).tolist()
    assert isinstance(grid.point_data['list'], np.ndarray)
    assert np.allclose(grid.point_data['list'], np.arange(grid.n_points))


def test_point_data_bad_value(grid):
    with pytest.raises(TypeError):
        grid.point_data['new_array'] = None

    with pytest.raises(ValueError):
        grid.point_data['new_array'] = np.arange(grid.n_points - 1)


def test_ipython_key_completions(grid):
    assert isinstance(grid._ipython_key_completions_(), list)


def test_cell_data(grid):
    key = 'test_array_cells'
    grid[key] = np.arange(grid.n_cells)
    assert key in grid.cell_data

    orig_value = grid.cell_data[key][0] / 1.0
    grid.cell_data[key][0] += 1
    assert orig_value == grid.cell_data[key][0] - 1

    del grid.cell_data[key]
    assert key not in grid.cell_data

    grid.cell_data[key] = np.arange(grid.n_cells)
    assert key in grid.cell_data

    assert np.allclose(grid[key], np.arange(grid.n_cells))

    grid.cell_data['list'] = np.arange(grid.n_cells).tolist()
    assert isinstance(grid.cell_data['list'], np.ndarray)
    assert np.allclose(grid.cell_data['list'], np.arange(grid.n_cells))


def test_cell_array_range(grid):
    rng = range(grid.n_cells)
    grid.cell_data['tmp'] = rng
    assert np.allclose(rng, grid.cell_data['tmp'])


def test_cell_data_bad_value(grid):
    with pytest.raises(TypeError):
        grid.cell_data['new_array'] = None

    with pytest.raises(ValueError):
        grid.cell_data['new_array'] = np.arange(grid.n_cells - 1)


def test_field_data(grid):
    key = 'test_array_field'
    # Add array of length not equal to n_cells or n_points
    n = grid.n_cells // 3
    grid.field_data[key] = np.arange(n)
    assert key in grid.field_data
    assert np.allclose(grid.field_data[key], np.arange(n))
    assert np.allclose(grid[key], np.arange(n))

    orig_value = grid.field_data[key][0] / 1.0
    grid.field_data[key][0] += 1
    assert orig_value == grid.field_data[key][0] - 1

    assert key in grid.array_names

    del grid.field_data[key]
    assert key not in grid.field_data

    grid.field_data['list'] = np.arange(n).tolist()
    assert isinstance(grid.field_data['list'], np.ndarray)
    assert np.allclose(grid.field_data['list'], np.arange(n))

    foo = np.arange(n) * 5
    grid.add_field_data(foo, 'foo')
    assert isinstance(grid.field_data['foo'], np.ndarray)
    assert np.allclose(grid.field_data['foo'], foo)

    with pytest.raises(ValueError):
        grid.set_active_scalars('foo')


@pytest.mark.parametrize('field', (range(5), np.ones((3, 3))[:, 0]))
def test_add_field_data(grid, field):
    grid.add_field_data(field, 'foo')
    assert isinstance(grid.field_data['foo'], np.ndarray)
    assert np.allclose(grid.field_data['foo'], field)


def test_modify_field_data(grid):
    field = range(4)
    grid.add_field_data(range(5), 'foo')
    grid.add_field_data(field, 'foo')
    assert np.allclose(grid.field_data['foo'], field)

    field = range(8)
    grid.field_data['foo'] = field
    assert np.allclose(grid.field_data['foo'], field)


def test_active_scalars_cell(grid):
    grid.add_field_data(range(5), 'foo')
    del grid.point_data['sample_point_scalars']
    del grid.point_data['VTKorigID']
    assert grid.active_scalars_info[1] == 'sample_cell_scalars'


def test_field_data_bad_value(grid):
    with pytest.raises(TypeError):
        grid.field_data['new_array'] = None


def test_copy(grid):
    grid_copy = grid.copy(deep=True)
    grid_copy.points[0] = np.nan
    assert not np.any(np.isnan(grid.points[0]))

    grid_copy_shallow = grid.copy(deep=False)
    grid_copy.points[0] += 0.1
    assert np.all(grid_copy_shallow.points[0] == grid.points[0])


def test_copy_metadata(globe):
    """Ensure metadata is copied correctly."""
    globe.point_data['bitarray'] = np.zeros(globe.n_points, dtype=bool)
    globe.point_data['complex_data'] = np.zeros(globe.n_points, dtype=np.complex128)

    globe_shallow = globe.copy(deep=False)
    assert globe_shallow._active_scalars_info is globe._active_scalars_info
    assert globe_shallow._active_vectors_info is globe._active_vectors_info
    assert globe_shallow._active_tensors_info is globe._active_tensors_info
    assert globe_shallow.textures is globe.textures
    assert globe_shallow.point_data['bitarray'].dtype == np.bool_
    assert globe_shallow.point_data['complex_data'].dtype == np.complex128
    assert globe_shallow._association_bitarray_names is globe._association_bitarray_names
    assert globe_shallow._association_complex_names is globe._association_complex_names

    globe_deep = globe.copy(deep=True)
    assert globe_deep.textures is not globe.textures
    assert globe_deep._active_scalars_info is not globe._active_scalars_info
    assert globe_deep._active_vectors_info is not globe._active_vectors_info
    assert globe_deep._active_tensors_info is not globe._active_tensors_info
    assert globe_deep._active_scalars_info == globe._active_scalars_info
    assert globe_deep._active_vectors_info == globe._active_vectors_info
    assert globe_deep._active_tensors_info == globe._active_tensors_info
    assert globe_deep.textures == globe.textures
    assert globe_deep.point_data['bitarray'].dtype == np.bool_
    assert globe_deep.point_data['complex_data'].dtype == np.complex128
    assert (
        globe_deep._association_bitarray_names['POINT']
        is not globe._association_bitarray_names['POINT']
    )
    assert (
        globe_deep._association_complex_names['POINT']
        is not globe._association_complex_names['POINT']
    )

    globe.clear_textures()
    assert not globe.textures
    assert globe_deep.textures
    assert not globe_shallow.textures


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


def test_set_points():
    dataset = pyvista.UnstructuredGrid()
    points = np.random.random((10, 3))
    dataset.points = pyvista.vtk_points(points)


def test_translate_should_fail_bad_points_or_transform(grid):
    points = np.random.random((10, 2))
    bad_points = np.random.random((10, 2))
    trans = np.random.random((4, 4))
    bad_trans = np.random.random((2, 4))
    with pytest.raises(ValueError):
        pyvista.utilities.transformations.apply_transformation_to_points(trans, bad_points)

    with pytest.raises(ValueError):
        pyvista.utilities.transformations.apply_transformation_to_points(bad_trans, points)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=HYPOTHESIS_MAX_EXAMPLES,
)
@given(array=arrays(dtype=np.float32, shape=array_shapes(max_dims=5, max_side=5)))
def test_transform_should_fail_given_wrong_numpy_shape(array, grid):
    assume(array.shape != (4, 4))
    with pytest.raises(ValueError):
        grid.transform(array)


@pytest.mark.parametrize('axis_amounts', [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
def test_translate_should_translate_grid(grid, axis_amounts):
    grid_copy = grid.copy()
    grid_copy.translate(axis_amounts, inplace=True)

    grid_points = grid.points.copy() + np.array(axis_amounts)
    assert np.allclose(grid_copy.points, grid_points)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=HYPOTHESIS_MAX_EXAMPLES,
)
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
    getattr(grid_b, f'rotate_{axis}')(angle, inplace=True)
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
    grid.point_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.point_data['bool_arr'].all()
    assert grid.point_data['bool_arr'].all()
    assert grid.point_data['bool_arr'].dtype == np.bool_


def test_cells_np_bool(grid):
    bool_arr = np.zeros(grid.n_cells, np.bool_)
    grid.cell_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.cell_data['bool_arr'].all()
    assert grid.cell_data['bool_arr'].all()
    assert grid.cell_data['bool_arr'].dtype == np.bool_


def test_field_np_bool(grid):
    bool_arr = np.zeros(grid.n_cells // 3, np.bool_)
    grid.field_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert grid.field_data['bool_arr'].all()
    assert grid.field_data['bool_arr'].all()
    assert grid.field_data['bool_arr'].dtype == np.bool_


def test_cells_uint8(grid):
    arr = np.zeros(grid.n_cells, np.uint8)
    grid.cell_data['arr'] = arr
    arr[:] = np.arange(grid.n_cells)
    assert np.allclose(grid.cell_data['arr'], np.arange(grid.n_cells))


def test_points_uint8(grid):
    arr = np.zeros(grid.n_points, np.uint8)
    grid.point_data['arr'] = arr
    arr[:] = np.arange(grid.n_points)
    assert np.allclose(grid.point_data['arr'], np.arange(grid.n_points))


def test_field_uint8(grid):
    n = grid.n_points // 3
    arr = np.zeros(n, np.uint8)
    grid.field_data['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(grid.field_data['arr'], np.arange(n))


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
    assert np.allclose(grid.point_data['bint_arr'], np_array)


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
    assert np.allclose(grid.cell_data['bint_arr'], np_array)


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
    assert np.allclose(grid.field_data['bint_arr'], np_array)


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
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0.5, 0],
            [0, 0.5, 0],
        ]
    )

    # mesh faces
    faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]]).astype(np.int8)

    # Create simple texture coordinates
    t_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # Create the poly data
    mesh = pyvista.PolyData(vertices, faces)
    # Attempt setting the texture coordinates
    mesh.active_t_coords = t_coords
    # now grab the texture coordinates
    foo = mesh.active_t_coords
    assert np.allclose(foo, t_coords)
    texture = Texture(examples.mapfile)
    mesh.textures['map'] = texture
    assert mesh.textures['map'] is not None
    mesh.clear_textures()
    assert len(mesh.textures) == 0


def test_multiple_texture_coordinates():
    mesh = examples.load_airplane()
    mesh.texture_map_to_plane(inplace=True, name="tex_a", use_bounds=False)
    mesh.texture_map_to_plane(inplace=True, name="tex_b", use_bounds=True)
    assert not np.allclose(mesh["tex_a"], mesh["tex_b"])
    texture = Texture(examples.mapfile)
    mesh.textures["tex_a"] = texture.copy()
    mesh.textures["tex_b"] = texture.copy()
    mesh._activate_texture("tex_a")
    assert np.allclose(mesh.active_t_coords, mesh["tex_a"])
    mesh._activate_texture("tex_b")
    assert np.allclose(mesh.active_t_coords, mesh["tex_b"])

    # Now test copying
    cmesh = mesh.copy()
    assert len(cmesh.textures) == 2
    assert "tex_a" in cmesh.textures
    assert "tex_b" in cmesh.textures


def test_inplace_no_overwrite_texture_coordinates():
    mesh = pyvista.Box()
    truth = mesh.texture_map_to_plane(inplace=False)
    mesh.texture_map_to_sphere(inplace=True)
    test = mesh.texture_map_to_plane(inplace=True)
    assert np.allclose(truth.active_t_coords, test.active_t_coords)


def test_invalid_vector(grid):
    with pytest.raises(ValueError):
        grid["vectors"] = np.empty(10)

    with pytest.raises(ValueError):
        grid["vectors"] = np.empty((3, 2))

    with pytest.raises(ValueError):
        grid["vectors"] = np.empty((3, 3))


def test_no_t_coords(grid):
    assert grid.active_t_coords is None


def test_no_arrows(grid):
    assert grid.arrows is None


def test_arrows():
    sphere = pyvista.Sphere(radius=3.14)

    # make cool swirly pattern
    vectors = np.vstack(
        (np.sin(sphere.points[:, 0]), np.cos(sphere.points[:, 1]), np.cos(sphere.points[:, 2]))
    ).T

    # add and scales
    sphere["vectors"] = vectors * 0.3
    sphere.set_active_vectors("vectors")
    assert np.allclose(sphere.active_vectors, vectors * 0.3)
    assert np.allclose(sphere["vectors"], vectors * 0.3)

    assert sphere.active_vectors_info[1] == 'vectors'
    arrows = sphere.arrows
    assert isinstance(arrows, pyvista.PolyData)
    assert np.any(arrows.points)
    assert arrows.active_vectors_name == 'GlyphVector'


def active_component_consistency_check(grid, component_type, field_association="point"):
    """
    Tests if the active component (scalars, vectors, tensors) actually reflects the underlying VTK dataset
    """
    component_type = component_type.lower()
    vtk_component_type = component_type.capitalize()

    field_association = field_association.lower()
    vtk_field_association = field_association.capitalize()

    pv_arr = getattr(grid, "active_" + component_type)
    vtk_arr = getattr(
        getattr(grid, f"Get{vtk_field_association}Data")(), f"Get{vtk_component_type}"
    )()

    assert (pv_arr is None and vtk_arr is None) or np.allclose(pv_arr, vtk_to_numpy(vtk_arr))


def test_set_active_vectors(grid):
    vector_arr = np.arange(grid.n_points * 3).reshape([grid.n_points, 3])
    grid.point_data['vector_arr'] = vector_arr
    grid.active_vectors_name = 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")
    assert grid.active_vectors_name == 'vector_arr'
    assert np.allclose(grid.active_vectors, vector_arr)

    grid.active_vectors_name = None
    assert grid.active_vectors_name is None
    active_component_consistency_check(grid, "vectors", "point")


def test_set_active_tensors(grid):
    tensor_arr = np.arange(grid.n_points * 9).reshape([grid.n_points, 9])
    grid.point_data['tensor_arr'] = tensor_arr
    grid.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")
    assert grid.active_tensors_name == 'tensor_arr'
    assert np.allclose(grid.active_tensors, tensor_arr)

    grid.active_tensors_name = None
    assert grid.active_tensors_name is None
    active_component_consistency_check(grid, "tensors", "point")


def test_set_t_coords(grid):
    with pytest.raises(TypeError):
        grid.active_t_coords = [1, 2, 3]

    with pytest.raises(ValueError):
        grid.active_t_coords = np.empty(10)

    with pytest.raises(ValueError):
        grid.active_t_coords = np.empty((3, 3))

    with pytest.raises(ValueError):
        grid.active_t_coords = np.empty((grid.n_points, 1))


def test_activate_texture_none(grid):
    with pytest.warns(UserWarning, match=r'not a key'):
        assert grid._activate_texture('not a key') is None

    with pytest.warns(UserWarning, match=r'No textures associated'):
        assert grid._activate_texture(True) is None


def test_set_active_vectors_fail(grid):
    with pytest.raises(ValueError):
        grid.set_active_vectors('not a vector')

    active_component_consistency_check(grid, "vectors", "point")
    vector_arr = np.arange(grid.n_points * 3).reshape([grid.n_points, 3])
    grid.point_data['vector_arr'] = vector_arr
    grid.active_vectors_name = 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")

    grid.point_data['scalar_arr'] = np.zeros([grid.n_points])

    with pytest.raises(ValueError):
        grid.set_active_vectors('scalar_arr')

    assert grid.active_vectors_name == 'vector_arr'
    active_component_consistency_check(grid, "vectors", "point")


def test_set_active_tensors_fail(grid):
    with pytest.raises(ValueError):
        grid.set_active_tensors('not a tensor')

    active_component_consistency_check(grid, "tensors", "point")
    tensor_arr = np.arange(grid.n_points * 9).reshape([grid.n_points, 9])
    grid.point_data['tensor_arr'] = tensor_arr
    grid.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")

    grid.point_data['scalar_arr'] = np.zeros([grid.n_points])
    grid.point_data['vector_arr'] = np.zeros([grid.n_points, 3])

    with pytest.raises(ValueError):
        grid.set_active_tensors('scalar_arr')

    with pytest.raises(ValueError):
        grid.set_active_tensors('vector_arr')

    assert grid.active_tensors_name == 'tensor_arr'
    active_component_consistency_check(grid, "tensors", "point")


def test_set_active_scalars(grid):
    arr = np.arange(grid.n_cells)
    grid.cell_data['tmp'] = arr
    grid.set_active_scalars('tmp')
    assert np.allclose(grid.active_scalars, arr)
    # Make sure we can set no active scalars
    grid.set_active_scalars(None)
    assert grid.GetPointData().GetScalars() is None
    assert grid.GetCellData().GetScalars() is None


def test_set_active_scalars_name(grid):
    point_keys = list(grid.point_data.keys())
    grid.active_scalars_name = point_keys[0]
    grid.active_scalars_name = None


def test_rename_array_point(grid):
    point_keys = list(grid.point_data.keys())
    old_name = point_keys[0]
    new_name = 'point changed'
    grid.set_active_scalars(old_name, preference='point')
    grid.rename_array(old_name, new_name, preference='point')
    assert new_name in grid.point_data
    assert old_name not in grid.point_data
    assert new_name == grid.active_scalars_name


def test_rename_array_cell(grid):
    cell_keys = list(grid.cell_data.keys())
    old_name = cell_keys[0]
    new_name = 'cell changed'
    grid.rename_array(old_name, new_name)
    assert new_name in grid.cell_data
    assert old_name not in grid.cell_data


def test_rename_array_field(grid):
    grid.field_data['fieldfoo'] = np.array([8, 6, 7])
    field_keys = list(grid.field_data.keys())
    old_name = field_keys[0]
    new_name = 'cell changed'
    grid.rename_array(old_name, new_name)
    assert new_name in grid.field_data
    assert old_name not in grid.field_data


def test_change_name_fail(grid):
    with pytest.raises(KeyError):
        grid.rename_array('not a key', '')


def test_get_cell_array_fail():
    sphere = pyvista.Sphere()
    with pytest.raises(TypeError):
        sphere.cell_data[None]


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
    grid.cell_data['_cell_vectors'] = arr
    grid.set_active_vectors('_cell_vectors')
    assert grid.active_vectors_name == '_cell_vectors'
    assert np.allclose(grid.active_vectors, arr)


def test_axis_rotation_invalid():
    with pytest.raises(ValueError):
        pyvista.utilities.axis_rotation(np.empty((3, 3)), 0, False, axis='not')


def test_axis_rotation_not_inplace():
    p = np.eye(3)
    p_out = pyvista.utilities.axis_rotation(p, 1, False, axis='x')
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


def test_clear_data():
    # First try on an empty mesh
    grid = pyvista.UniformGrid(dimensions=(10, 10, 10))
    # Now try something more complicated
    grid.clear_data()
    grid['foo-p'] = np.random.rand(grid.n_points)
    grid['foo-c'] = np.random.rand(grid.n_cells)
    grid.field_data['foo-f'] = np.random.rand(grid.n_points * grid.n_cells)
    assert grid.n_arrays == 3
    grid.clear_data()
    assert grid.n_arrays == 0


def test_scalars_dict_update():
    mesh = examples.load_uniform()
    n = len(mesh.point_data)
    arrays = {'foo': np.arange(mesh.n_points), 'rand': np.random.random(mesh.n_points)}
    mesh.point_data.update(arrays)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_data) == n + 2

    # Test update from Table
    table = pyvista.Table(arrays)
    mesh = examples.load_uniform()
    mesh.point_data.update(table)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_data) == n + 2


def test_handle_array_with_null_name():
    poly = pyvista.PolyData()
    # Add point array with no name
    poly.GetPointData().AddArray(pyvista.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    pdata = poly.point_data
    assert pdata is not None
    assert len(pdata) == 1
    # Add cell array with no name
    poly.GetCellData().AddArray(pyvista.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    cdata = poly.cell_data
    assert cdata is not None
    assert len(cdata) == 1
    # Add field array with no name
    poly.GetFieldData().AddArray(pyvista.convert_array(np.array([5, 6])))
    html = poly._repr_html_()
    assert html is not None
    fdata = poly.field_data
    assert fdata is not None
    assert len(fdata) == 1


def test_add_point_array_list(grid):
    rng = range(grid.n_points)
    grid.point_data['tmp'] = rng
    assert np.allclose(grid.point_data['tmp'], rng)


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
    index = mesh.find_closest_cell(node)
    assert isinstance(index, int)


def test_find_closest_cells():
    mesh = pyvista.Sphere()
    # simply get the face centers, ordered by cell Id
    fcent = mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]].mean(1)
    fcent_copy = fcent.copy()
    indices = mesh.find_closest_cell(fcent)

    # Make sure we match the face centers
    assert np.allclose(indices, np.arange(mesh.n_faces))

    # Make sure arg was not modified
    assert np.array_equal(fcent, fcent_copy)


def test_find_closest_cell_surface_point():
    mesh = pyvista.Rectangle()

    point = np.array([0.5, 0.5, -1.0])
    point2 = np.array([1.0, 1.0, -1.0])
    points = np.vstack((point, point2))

    _, closest_point = mesh.find_closest_cell(point, return_closest_point=True)
    assert np.allclose(closest_point, [0.5, 0.5, 0])

    _, closest_points = mesh.find_closest_cell(points, return_closest_point=True)
    assert np.allclose(closest_points, [[0.5, 0.5, 0], [1.0, 1.0, 0]])


def test_find_containing_cell():
    mesh = pyvista.UniformGrid(dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0])
    node = np.array([0.3, 0.3, 0.0])
    index = mesh.find_containing_cell(node)
    assert index == 5


def test_find_containing_cells():
    mesh = pyvista.UniformGrid(dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0])
    points = np.array([[0.3, 0.3, 0], [0.6, 0.6, 0]])
    points_copy = points.copy()
    indices = mesh.find_containing_cell(points)
    assert np.allclose(indices, [5, 10])
    assert np.array_equal(points, points_copy)


def test_find_cells_along_line():
    mesh = pyvista.Cube()
    indices = mesh.find_cells_along_line([0, 0, -1], [0, 0, 1])
    assert len(indices) == 2


def test_find_cells_within_bounds():
    mesh = pyvista.Cube()

    bounds = [
        mesh.bounds[0] * 2.0,
        mesh.bounds[1] * 2.0,
        mesh.bounds[2] * 2.0,
        mesh.bounds[3] * 2.0,
        mesh.bounds[4] * 2.0,
        mesh.bounds[5] * 2.0,
    ]
    indices = mesh.find_cells_within_bounds(bounds)
    assert len(indices) == mesh.n_cells

    bounds = [
        mesh.bounds[0] * 0.5,
        mesh.bounds[1] * 0.5,
        mesh.bounds[2] * 0.5,
        mesh.bounds[3] * 0.5,
        mesh.bounds[4] * 0.5,
        mesh.bounds[5] * 0.5,
    ]
    indices = mesh.find_cells_within_bounds(bounds)
    assert len(indices) == 0


def test_setting_points_by_different_types(grid):
    grid_copy = grid.copy()
    grid.points = grid_copy.points
    assert np.array_equal(grid.points, grid_copy.points)

    grid.points = np.array(grid_copy.points)
    assert np.array_equal(grid.points, grid_copy.points)

    grid.points = grid_copy.points.tolist()
    assert np.array_equal(grid.points, grid_copy.points)

    pgrid = pyvista.PolyData([0.0, 0.0, 0.0])
    pgrid.points = [1.0, 1.0, 1.0]
    assert np.array_equal(pgrid.points, [[1.0, 1.0, 1.0]])

    pgrid.points = np.array([2.0, 2.0, 2.0])
    assert np.array_equal(pgrid.points, [[2.0, 2.0, 2.0]])


def test_empty_points():
    pdata = pyvista.PolyData()
    assert np.allclose(pdata.points, np.empty(3))


def test_no_active():
    pdata = pyvista.PolyData()
    assert pdata.active_scalars is None

    with pytest.raises(TypeError):
        pdata.point_data[None]


def test_get_data_range(grid):
    # Test with blank mesh
    mesh = pyvista.Sphere()
    mesh.clear_data()
    rng = mesh.get_data_range()
    assert all(np.isnan(rng))
    with pytest.raises(KeyError):
        rng = mesh.get_data_range('some data')

    # Test with some data
    grid.active_scalars_name = 'sample_point_scalars'
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
    assert len(copy.field_data) == 0
    assert len(copy.cell_data) == 0
    assert len(copy.point_data) == 0


def test_copy_attributes(grid):
    classname = grid.__class__.__name__
    copy = eval(f'pyvista.{classname}')()
    copy.copy_attributes(grid)
    assert copy.n_cells == 0
    assert copy.n_points == 0
    assert copy.field_data.keys() == grid.field_data.keys()
    assert copy.cell_data.keys() == grid.cell_data.keys()
    assert copy.point_data.keys() == grid.point_data.keys()


def test_cell_n_points(grid):
    with pytest.warns(PyVistaDeprecationWarning):
        npoints = grid.cell_n_points(0)
    if pyvista._version.version_info >= (0, 42, 0):
        raise RuntimeError('Remove this deprecated method')
    assert isinstance(npoints, int)
    assert npoints >= 0


def test_cell_points(grid):
    with pytest.warns(PyVistaDeprecationWarning):
        points = grid.cell_points(0)
    if pyvista._version.version_info >= (0, 42, 0):
        raise RuntimeError('Remove this deprecated method')
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3


def test_cell_point_ids(grid):
    with pytest.warns(PyVistaDeprecationWarning):
        point_ids = grid.cell_point_ids(0)
    if pyvista._version.version_info >= (0, 42, 0):
        raise RuntimeError('Remove this deprecated method')
    assert isinstance(point_ids, list)
    with pytest.warns(PyVistaDeprecationWarning):
        assert len(point_ids) == grid.cell_n_points(0)
    assert all([isinstance(id, int) for id in point_ids])
    assert all([0 <= id < grid.n_points for id in point_ids])


def test_cell_bounds(grid):
    with pytest.warns(PyVistaDeprecationWarning):
        bounds = grid.cell_bounds(0)
    if pyvista._version.version_info >= (0, 42, 0):
        raise RuntimeError('Remove this deprecated method')
    assert isinstance(bounds, tuple)
    assert len(bounds) == 6


def test_cell_type(grid):
    with pytest.warns(PyVistaDeprecationWarning):
        ctype = grid.cell_type(0)
    if pyvista._version.version_info >= (0, 42, 0):
        raise RuntimeError('Remove this deprecated method')
    assert isinstance(ctype, int)


def test_point_is_inside_cell():
    grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
    assert grid.point_is_inside_cell(0, [0.5, 0.5, 0.5])
    assert not grid.point_is_inside_cell(0, [-0.5, -0.5, -0.5])

    assert grid.point_is_inside_cell(0, np.array([0.5, 0.5, 0.5]))

    # cell ind out of range
    with pytest.raises(ValueError):
        grid.point_is_inside_cell(100000, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        grid.point_is_inside_cell(-1, [0.5, 0.5, 0.5])

    # cell ind wrong type
    with pytest.raises(TypeError):
        grid.point_is_inside_cell(0.1, [0.5, 0.5, 0.5])

    # point not well formed
    with pytest.raises(TypeError):
        grid.point_is_inside_cell(0, 0.5)
    with pytest.raises(ValueError):
        grid.point_is_inside_cell(0, [0.5, 0.5])

    # multi-dimensional
    in_cell = grid.point_is_inside_cell(0, [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    assert np.array_equal(in_cell, np.array([True, False]))


@pytest.mark.parametrize('pickle_format', ['xml', 'legacy'])
def test_serialize_deserialize(datasets, pickle_format):
    pyvista.set_pickle_format(pickle_format)
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

        for name in dataset.point_data:
            arr_have = dataset_2.point_data[name]
            arr_expected = dataset.point_data[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.cell_data:
            arr_have = dataset_2.cell_data[name]
            arr_expected = dataset.cell_data[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.field_data:
            arr_have = dataset_2.field_data[name]
            arr_expected = dataset.field_data[name]
            assert arr_have == pytest.approx(arr_expected)


def n_points(dataset):
    # used in multiprocessing test
    return dataset.n_points


@pytest.mark.parametrize('pickle_format', ['xml', 'legacy'])
def test_multiprocessing(datasets, pickle_format):
    # exercise pickling via multiprocessing
    pyvista.set_pickle_format(pickle_format)
    with multiprocessing.Pool(2) as p:
        res = p.map(n_points, datasets)
    for res, dataset in zip(res, datasets):
        assert res == dataset.n_points


def test_rotations_should_match_by_a_360_degree_difference():
    mesh = examples.load_airplane()

    point = np.random.random(3) - 0.5
    angle = (np.random.random() - 0.5) * 360.0
    vector = np.random.random(3) - 0.5

    # Rotate about x axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_x(angle=angle, point=point, inplace=True)
    rot2.rotate_x(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about y axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_y(angle=angle, point=point, inplace=True)
    rot2.rotate_y(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about z axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_z(angle=angle, point=point, inplace=True)
    rot2.rotate_z(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about custom vector.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_vector(vector=vector, angle=angle, point=point, inplace=True)
    rot2.rotate_vector(vector=vector, angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)


def test_rotate_x():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_x(30)
    assert isinstance(out, pyvista.StructuredGrid)
    with pytest.raises(TypeError):
        out = mesh.rotate_x(30, point=5)
    with pytest.raises(ValueError):
        out = mesh.rotate_x(30, point=[1, 3])


def test_rotate_y():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_y(30)
    assert isinstance(out, pyvista.StructuredGrid)
    with pytest.raises(TypeError):
        out = mesh.rotate_y(30, point=5)
    with pytest.raises(ValueError):
        out = mesh.rotate_y(30, point=[1, 3])


def test_rotate_z():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_z(30)
    assert isinstance(out, pyvista.StructuredGrid)
    with pytest.raises(TypeError):
        out = mesh.rotate_z(30, point=5)
    with pytest.raises(ValueError):
        out = mesh.rotate_z(30, point=[1, 3])


def test_rotate_vector():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_vector([1, 1, 1], 33)
    assert isinstance(out, pyvista.StructuredGrid)
    with pytest.raises(ValueError):
        out = mesh.rotate_vector([1, 1], 33)
    with pytest.raises(TypeError):
        out = mesh.rotate_vector(30, 33)


def test_transform_integers():
    # regression test for gh-1943
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    # build vtkPolyData from scratch to enforce int data
    poly = vtk.vtkPolyData()
    poly.SetPoints(pyvista.vtk_points(points))
    poly = pyvista.wrap(poly)
    poly.verts = [1, 0, 1, 1, 1, 2]
    # define active and inactive vectors with int values
    for dataset_attrs in poly.point_data, poly.cell_data:
        for key in 'active_v', 'inactive_v', 'active_n', 'inactive_n':
            dataset_attrs[key] = poly.points
        dataset_attrs.active_vectors_name = 'active_v'
        dataset_attrs.active_normals_name = 'active_n'

    # active vectors and normals should be converted by default
    for key in 'active_v', 'inactive_v', 'active_n', 'inactive_n':
        assert poly.point_data[key].dtype == np.int_
        assert poly.cell_data[key].dtype == np.int_

    with pytest.warns(UserWarning):
        poly.rotate_x(angle=10, inplace=True)

    # check that points were converted and transformed correctly
    assert poly.points.dtype == np.float32
    assert poly.points[-1, 1] != 0
    # assert that exactly active vectors and normals were converted
    for key in 'active_v', 'active_n':
        assert poly.point_data[key].dtype == np.float32
        assert poly.cell_data[key].dtype == np.float32
    for key in 'inactive_v', 'inactive_n':
        assert poly.point_data[key].dtype == np.int_
        assert poly.cell_data[key].dtype == np.int_


@pytest.mark.xfail(reason='VTK bug')
def test_transform_integers_vtkbug_present():
    # verify that the VTK transform bug is still there
    # if this test starts to pass, we can remove the
    # automatic float conversion from ``DataSet.transform``
    # along with this test
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    # build vtkPolyData from scratch to enforce int data
    poly = vtk.vtkPolyData()
    poly.SetPoints(pyvista.vtk_points(points))

    # manually put together a rotate_x(10) transform
    trans_arr = pyvista.transformations.axis_angle_rotation((1, 0, 0), 10, deg=True)
    trans_mat = pyvista.vtkmatrix_from_array(trans_arr)
    trans = vtk.vtkTransform()
    trans.SetMatrix(trans_mat)
    trans_filt = vtk.vtkTransformFilter()
    trans_filt.SetInputDataObject(poly)
    trans_filt.SetTransform(trans)
    trans_filt.Update()
    poly = pyvista.wrap(trans_filt.GetOutputDataObject(0))
    # the bug is that e.g. 0.98 gets truncated to 0
    assert poly.points[-1, 1] != 0


def test_scale():
    mesh = examples.load_airplane()

    xyz = np.random.random(3)
    scale1 = mesh.copy()
    scale2 = mesh.copy()
    scale1.scale(xyz, inplace=True)
    scale2.points *= xyz
    scale3 = mesh.scale(xyz, inplace=False)
    assert np.allclose(scale1.points, scale2.points)
    assert np.allclose(scale3.points, scale2.points)
    # test scalar scale case
    scale1 = mesh.copy()
    scale2 = mesh.copy()
    xyz = 4.0
    scale1.scale(xyz, inplace=True)
    scale2.scale([xyz] * 3, inplace=True)
    assert np.allclose(scale1.points, scale2.points)
    # test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.scale(xyz)
    assert isinstance(out, pyvista.StructuredGrid)


def test_flip_x():
    mesh = examples.load_airplane()
    flip_x1 = mesh.copy()
    flip_x2 = mesh.copy()
    flip_x1.flip_x(point=(0, 0, 0), inplace=True)
    flip_x2.points[:, 0] *= -1.0
    assert np.allclose(flip_x1.points, flip_x2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_x()
    assert isinstance(out, pyvista.StructuredGrid)


def test_flip_y():
    mesh = examples.load_airplane()
    flip_y1 = mesh.copy()
    flip_y2 = mesh.copy()
    flip_y1.flip_y(point=(0, 0, 0), inplace=True)
    flip_y2.points[:, 1] *= -1.0
    assert np.allclose(flip_y1.points, flip_y2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_y()
    assert isinstance(out, pyvista.StructuredGrid)


def test_flip_z():
    mesh = examples.load_airplane()
    flip_z1 = mesh.copy()
    flip_z2 = mesh.copy()
    flip_z1.flip_z(point=(0, 0, 0), inplace=True)
    flip_z2.points[:, 2] *= -1.0
    assert np.allclose(flip_z1.points, flip_z2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_z()
    assert isinstance(out, pyvista.StructuredGrid)


def test_flip_normal():
    mesh = examples.load_airplane()
    flip_normal1 = mesh.copy()
    flip_normal2 = mesh.copy()
    flip_normal1.flip_normal(normal=[1.0, 0.0, 0.0], inplace=True)
    flip_normal2.flip_x(inplace=True)
    assert np.allclose(flip_normal1.points, flip_normal2.points)

    flip_normal3 = mesh.copy()
    flip_normal4 = mesh.copy()
    flip_normal3.flip_normal(normal=[0.0, 1.0, 0.0], inplace=True)
    flip_normal4.flip_y(inplace=True)
    assert np.allclose(flip_normal3.points, flip_normal4.points)

    flip_normal5 = mesh.copy()
    flip_normal6 = mesh.copy()
    flip_normal5.flip_normal(normal=[0.0, 0.0, 1.0], inplace=True)
    flip_normal6.flip_z(inplace=True)
    assert np.allclose(flip_normal5.points, flip_normal6.points)

    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_normal(normal=[1.0, 0.0, 0.5])
    assert isinstance(out, pyvista.StructuredGrid)


def test_active_normals(sphere):
    # both cell and point normals
    mesh = sphere.compute_normals()
    assert mesh.active_normals.shape[0] == mesh.n_points

    mesh = sphere.compute_normals(point_normals=False)
    assert mesh.active_normals.shape[0] == mesh.n_cells


@pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)
def test_cast_to_pointset(sphere):
    sphere = sphere.elevation()
    pointset = sphere.cast_to_pointset()
    assert isinstance(pointset, pyvista.PointSet)

    assert not np.may_share_memory(sphere.points, pointset.points)
    assert not np.may_share_memory(sphere.active_scalars, pointset.active_scalars)
    assert np.allclose(sphere.points, pointset.points)
    assert np.allclose(sphere.active_scalars, pointset.active_scalars)

    pointset.points[:] = 0
    assert not np.allclose(sphere.points, pointset.points)

    pointset.active_scalars[:] = 0
    assert not np.allclose(sphere.active_scalars, pointset.active_scalars)


@pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)
def test_cast_to_pointset_implicit(uniform):
    pointset = uniform.cast_to_pointset(pass_cell_data=True)
    assert isinstance(pointset, pyvista.PointSet)
    assert pointset.n_arrays == uniform.n_arrays

    assert not np.may_share_memory(uniform.active_scalars, pointset.active_scalars)
    assert np.allclose(uniform.active_scalars, pointset.active_scalars)

    ctp = uniform.cell_data_to_point_data()
    for name in ctp.point_data.keys():
        assert np.allclose(ctp[name], pointset[name])

    for i, name in enumerate(uniform.point_data.keys()):
        pointset[name][:] = i
        assert not np.allclose(uniform[name], pointset[name])


def test_cast_to_poly_points_implicit(uniform):
    points = uniform.cast_to_poly_points(pass_cell_data=True)
    assert isinstance(points, pyvista.PolyData)
    assert points.n_arrays == uniform.n_arrays
    assert len(points.cell_data) == len(uniform.cell_data)
    assert len(points.point_data) == len(uniform.point_data)

    assert not np.may_share_memory(uniform.active_scalars, points.active_scalars)
    assert np.allclose(uniform.active_scalars, points.active_scalars)

    ctp = uniform.cell_data_to_point_data()
    for name in ctp.point_data.keys():
        assert np.allclose(ctp[name], points[name])

    for i, name in enumerate(uniform.point_data.keys()):
        points[name][:] = i
        assert not np.allclose(uniform[name], points[name])


def test_partition(hexbeam):
    if pyvista.vtk_version_info < (9, 1, 0):
        with pytest.raises(VTKVersionError):
            hexbeam.partition(2)
        return
    # split as composite
    n_part = 2
    out = hexbeam.partition(n_part)
    assert isinstance(out, pyvista.MultiBlock)
    assert len(out) == 2

    # split as unstrucutred grid
    out = hexbeam.partition(hexbeam.n_cells, as_composite=False)
    assert isinstance(hexbeam, pyvista.UnstructuredGrid)
    assert out.n_points > hexbeam.n_points


def test_explode(datasets):
    for dataset in datasets:
        out = dataset.explode()
        assert out.n_cells == dataset.n_cells
        assert out.n_points > dataset.n_points


def test_separate_cells(hexbeam):
    assert hexbeam.n_points != hexbeam.n_cells * 8
    sep_grid = hexbeam.separate_cells()
    assert sep_grid.n_points == hexbeam.n_cells * 8


def test_volume_area():
    def assert_volume(grid):
        assert np.isclose(grid.volume, 64.0)
        assert np.isclose(grid.area, 0.0)

    def assert_area(grid):
        assert np.isclose(grid.volume, 0.0)
        assert np.isclose(grid.area, 16.0)

    # UniformGrid 3D size 4x4x4
    vol_grid = pyvista.UniformGrid(dimensions=(5, 5, 5))
    assert_volume(vol_grid)

    # 2D grid size 4x4
    surf_grid = pyvista.UniformGrid(dimensions=(5, 5, 1))
    assert_area(surf_grid)

    # UnstructuredGrid
    assert_volume(vol_grid.cast_to_unstructured_grid())
    assert_area(surf_grid.cast_to_unstructured_grid())

    # StructuredGrid
    assert_volume(vol_grid.cast_to_structured_grid())
    assert_area(surf_grid.cast_to_structured_grid())

    # Rectilinear
    assert_volume(vol_grid.cast_to_rectilinear_grid())
    assert_area(surf_grid.cast_to_rectilinear_grid())

    # PolyData
    # cube of size 4
    # PolyData is special because it is a 2D surface that can enclose a volume
    grid = pyvista.UniformGrid(dimensions=(5, 5, 5)).extract_surface()
    assert np.isclose(grid.volume, 64.0)
    assert np.isclose(grid.area, 96.0)


# ------------------
# Connectivity tests
# ------------------

i0s = [0, 1]
grids = [
    load_airplane(),
    load_structured(),
    load_hexbeam(),
    load_rectilinear(),
    load_tetbeam(),
    load_uniform(),
    load_explicit_structured(),
]
grids_cells = grids[:-1]

ids = list(map(type, grids))
ids_cells = list(map(type, grids_cells))


def test_raises_cell_neighbors_ExplicitStructuredGrid(datasets_vtk9):
    for dataset in datasets_vtk9:
        with pytest.raises(TypeError):
            _ = dataset.cell_neighbors(0)


def test_raises_point_neighbors_ind_overflow(grid):
    with pytest.raises(IndexError):
        _ = grid.point_neighbors(grid.n_points)


def test_raises_cell_neighbors_connections(grid):
    with pytest.raises(ValueError, match='got "topological"'):
        _ = grid.cell_neighbors(0, "topological")


@pytest.mark.parametrize("grid", grids, ids=ids)
@pytest.mark.parametrize("i0", i0s)
def test_point_cell_ids(grid: DataSet, i0):
    cell_ids = grid.point_cell_ids(i0)

    assert isinstance(cell_ids, list)
    assert all([isinstance(id, int) for id in cell_ids])
    assert all([0 <= id < grid.n_cells for id in cell_ids])
    assert len(cell_ids) > 0

    # Check that the output cells contain the i0-th point but also that the
    # remaining cells does not contain this point id
    for c in cell_ids:
        assert i0 in grid.get_cell(c).point_ids

    others = [i for i in range(grid.n_cells) if i not in cell_ids]
    for c in others:
        assert i0 not in grid.get_cell(c).point_ids


@pytest.mark.parametrize("grid", grids_cells, ids=ids_cells)
@pytest.mark.parametrize("i0", i0s)
def test_cell_point_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, "points")
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all([isinstance(id, int) for id in cell_ids])
    assert all([0 <= id < grid.n_cells for id in cell_ids])
    assert len(cell_ids) > 0

    # Check that all the neighbors cells share at least one point with the
    # current cell
    current_points = set(cell.point_ids)
    for i in cell_ids:
        neighbor_points = set(grid.get_cell(i).point_ids)
        assert not neighbor_points.isdisjoint(current_points)

    # Check that other cells do not share a point with the current cell
    other_ids = [i for i in range(grid.n_cells) if (i not in cell_ids and i != i0)]
    for i in other_ids:
        neighbor_points = set(grid.get_cell(i).point_ids)
        assert neighbor_points.isdisjoint(current_points)


@pytest.mark.parametrize("grid", grids_cells, ids=ids_cells)
@pytest.mark.parametrize("i0", i0s)
def test_cell_edge_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, "edges")
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all([isinstance(id, int) for id in cell_ids])
    assert all([0 <= id < grid.n_cells for id in cell_ids])
    assert len(cell_ids) > 0

    # Check that all the neighbors cells share at least one edge with the
    # current cell
    current_points = set()
    for e in cell.edges:
        current_points.add(frozenset(e.point_ids))

    for i in cell_ids:
        neighbor_points = set()
        neighbor_cell = grid.get_cell(i)

        for ie in range(neighbor_cell.n_edges):
            e = neighbor_cell.get_edge(ie)
            neighbor_points.add(frozenset(e.point_ids))

        assert not neighbor_points.isdisjoint(current_points)

    # Check that other cells do not share an edge with the current cell
    other_ids = [i for i in range(grid.n_cells) if (i not in cell_ids and i != i0)]
    for i in other_ids:
        neighbor_points = set()
        neighbor_cell = grid.get_cell(i)

        for ie in range(neighbor_cell.n_edges):
            e = neighbor_cell.get_edge(ie)
            neighbor_points.add(frozenset(e.point_ids))

        assert neighbor_points.isdisjoint(current_points)


# Slice grids since some do not contain faces
@pytest.mark.parametrize("grid", grids_cells[2:], ids=ids_cells[2:])
@pytest.mark.parametrize("i0", i0s)
def test_cell_face_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, "faces")
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all([isinstance(id, int) for id in cell_ids])
    assert all([0 <= id < grid.n_cells for id in cell_ids])
    assert len(cell_ids) > 0

    # Check that all the neighbors cells share at least one face with the
    # current cell
    current_points = set()
    for f in cell.faces:
        current_points.add(frozenset(f.point_ids))

    for i in cell_ids:
        neighbor_points = set()
        neighbor_cell = grid.get_cell(i)

        for ifa in range(neighbor_cell.n_faces):
            f = neighbor_cell.get_face(ifa)
            neighbor_points.add(frozenset(f.point_ids))

        assert not neighbor_points.isdisjoint(current_points)

    # Check that other cells do not share a face with the current cell
    other_ids = [i for i in range(grid.n_cells) if (i not in cell_ids and i != i0)]
    for i in other_ids:
        neighbor_points = set()
        neighbor_cell = grid.get_cell(i)

        for ifa in range(neighbor_cell.n_faces):
            f = neighbor_cell.get_face(ifa)
            neighbor_points.add(frozenset(f.point_ids))

        assert neighbor_points.isdisjoint(current_points)


@pytest.mark.parametrize("grid", grids_cells, ids=ids_cells)
@pytest.mark.parametrize("i0", i0s, ids=lambda x: f"i0={x}")
@pytest.mark.parametrize("n_levels", [1, 3], ids=lambda x: f"n_levels={x}")
@pytest.mark.parametrize(
    "connections", ["points", "edges", "faces"], ids=lambda x: f"connections={x}"
)
def test_cell_neighbors_levels(grid: DataSet, i0, n_levels, connections):
    cell_ids = grid.cell_neighbors_levels(i0, connections=connections, n_levels=n_levels)

    if connections == "faces" and grid.get_cell(i0).dimension != 3:
        pytest.skip("Grid's cells does not contain faces")

    if n_levels == 1:
        # Consume generator and check length and consistency
        # with underlying method
        cell_ids = list(cell_ids)
        assert len(cell_ids) == 1
        cell_ids = cell_ids[0]
        assert len(cell_ids) > 0
        assert set(cell_ids) == set(grid.cell_neighbors(i0, connections=connections))

    else:
        for i, ids in enumerate(cell_ids):
            assert isinstance(ids, list)
            assert all([isinstance(id, int) for id in ids])
            assert all([0 <= id < grid.n_cells for id in ids])
            assert len(ids) > 0
        assert i == n_levels - 1


@pytest.mark.parametrize("grid", grids, ids=ids)
@pytest.mark.parametrize("i0", i0s)
@pytest.mark.parametrize("n_levels", [1, 3])
def test_point_neighbors_levels(grid: DataSet, i0, n_levels):
    point_ids = grid.point_neighbors_levels(i0, n_levels=n_levels)

    if n_levels == 1:
        # Consume generator and check length and consistency
        # with underlying method
        point_ids = list(point_ids)
        assert len(point_ids) == 1
        point_ids = point_ids[0]
        assert len(point_ids) > 0
        assert set(point_ids) == set(grid.point_neighbors(i0))

    else:
        for i, ids in enumerate(point_ids):
            assert isinstance(ids, list)
            assert all([isinstance(id, int) for id in ids])
            assert all([0 <= id < grid.n_points for id in ids])
            assert len(ids) > 0
        assert i == n_levels - 1
