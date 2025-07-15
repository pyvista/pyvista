"""Tests for pyvista.core.dataset."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista as pv
from pyvista import examples
from pyvista.core import dataset
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.examples import load_airplane
from pyvista.examples import load_explicit_structured
from pyvista.examples import load_hexbeam
from pyvista.examples import load_rectilinear
from pyvista.examples import load_structured
from pyvista.examples import load_tetbeam
from pyvista.examples import load_uniform

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pyvista.core.dataset import DataSet


def test_invalid_copy_from(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.copy_from(pv.Plane())


def test_memory_address(hexbeam):
    assert isinstance(hexbeam.memory_address, str)
    assert 'Addr' in hexbeam.memory_address


def test_point_data(hexbeam):
    key = 'test_array_points'
    hexbeam[key] = np.arange(hexbeam.n_points)
    assert key in hexbeam.point_data

    orig_value = hexbeam.point_data[key][0] / 1.0
    hexbeam.point_data[key][0] += 1
    assert orig_value == hexbeam.point_data[key][0] - 1

    del hexbeam.point_data[key]
    assert key not in hexbeam.point_data

    hexbeam.point_data[key] = np.arange(hexbeam.n_points)
    assert key in hexbeam.point_data

    assert np.allclose(hexbeam[key], np.arange(hexbeam.n_points))

    hexbeam.clear_point_data()
    assert len(hexbeam.point_data.keys()) == 0

    hexbeam.point_data['list'] = np.arange(hexbeam.n_points).tolist()
    assert isinstance(hexbeam.point_data['list'], np.ndarray)
    assert np.allclose(hexbeam.point_data['list'], np.arange(hexbeam.n_points))


def test_point_data_bad_value(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.point_data['new_array'] = None

    match = (
        "Invalid array shape. Array 'new_array' has length (98) but a length of (99) was expected."
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        hexbeam.point_data['new_array'] = np.arange(hexbeam.n_points - 1)


def test_ipython_key_completions(hexbeam):
    assert isinstance(hexbeam._ipython_key_completions_(), list)


def test_cell_data(hexbeam):
    key = 'test_array_cells'
    hexbeam[key] = np.arange(hexbeam.n_cells)
    assert key in hexbeam.cell_data

    orig_value = hexbeam.cell_data[key][0] / 1.0
    hexbeam.cell_data[key][0] += 1
    assert orig_value == hexbeam.cell_data[key][0] - 1

    del hexbeam.cell_data[key]
    assert key not in hexbeam.cell_data

    hexbeam.cell_data[key] = np.arange(hexbeam.n_cells)
    assert key in hexbeam.cell_data

    assert np.allclose(hexbeam[key], np.arange(hexbeam.n_cells))

    hexbeam.cell_data['list'] = np.arange(hexbeam.n_cells).tolist()
    assert isinstance(hexbeam.cell_data['list'], np.ndarray)
    assert np.allclose(hexbeam.cell_data['list'], np.arange(hexbeam.n_cells))


def test_cell_array_range(hexbeam):
    rng = range(hexbeam.n_cells)
    hexbeam.cell_data['tmp'] = rng
    assert np.allclose(rng, hexbeam.cell_data['tmp'])


def test_cell_data_bad_value(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.cell_data['new_array'] = None

    match = (
        "Invalid array shape. Array 'new_array' has length (39) but a length of (40) was expected."
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        hexbeam.cell_data['new_array'] = np.arange(hexbeam.n_cells - 1)


@pytest.mark.parametrize('empty_shape', [(0,), (-1, 0), (0, -1), (0, 0)])
@pytest.mark.parametrize('attribute', ['point_data', 'cell_data', 'field_data'])
@pytest.mark.parametrize('mesh_is_empty', [True, False])
def test_point_cell_field_data_empty_array(uniform, attribute, empty_shape, mesh_is_empty):
    # Test that setting empty arrays is only allowed when the mesh is
    # empty OR when setting field data.
    # Empty arrays with non-zero shape values are never allowed.

    mesh = pv.PolyData() if mesh_is_empty else uniform

    # Define empty array with a shape that matches the dataset
    if attribute == 'point_data':
        mesh_data_length = mesh.n_points
    elif attribute == 'cell_data':
        mesh_data_length = mesh.n_cells
    else:
        # Use an arbitrary non-zero positive value for field data in the non-empty case
        mesh_data_length = 0 if mesh_is_empty else 10
    if mesh_is_empty:
        assert mesh_data_length == 0
    else:
        assert mesh_data_length > 0

    # Replace `-1` in empty shape with the actual length of the mesh data
    empty_shape = np.array(empty_shape)
    empty_shape[empty_shape == -1] = mesh_data_length
    empty_shape = tuple(empty_shape.tolist())

    empty_array = np.ones(empty_shape)
    assert empty_array.size == 0
    assert empty_array.shape == empty_shape

    # Test setting the array
    data = getattr(mesh, attribute)
    if empty_shape in [(0,), (0, 0)] and (attribute == 'field_data' or mesh_is_empty):
        # Special case, no error raised
        data['new_array'] = empty_array
        assert 'new_array' in data
        assert data['new_array'].size == 0
        # Note: the output shape is always (0,) and may not match the input shape (bug?)
        assert data['new_array'].shape == (0,)
    else:
        # Expect error for all other cases
        with pytest.raises(ValueError, match='Invalid array shape.'):
            data['new_array'] = empty_array


def test_point_cell_data_single_scalar_no_exception_raised():
    try:
        m = pv.PolyData([0, 0, 0.0])
        m.point_data['foo'] = 1
        m.cell_data['bar'] = 1
        m['baz'] = 1
    except Exception as e:
        pytest.fail(f'Unexpected exception raised: {e}')


def test_field_data(hexbeam):
    key = 'test_array_field'
    # Add array of length not equal to n_cells or n_points
    n = hexbeam.n_cells // 3
    hexbeam.field_data[key] = np.arange(n)
    assert key in hexbeam.field_data
    assert np.allclose(hexbeam.field_data[key], np.arange(n))
    assert np.allclose(hexbeam[key], np.arange(n))

    orig_value = hexbeam.field_data[key][0] / 1.0
    hexbeam.field_data[key][0] += 1
    assert orig_value == hexbeam.field_data[key][0] - 1

    assert key in hexbeam.array_names

    del hexbeam.field_data[key]
    assert key not in hexbeam.field_data

    hexbeam.field_data['list'] = np.arange(n).tolist()
    assert isinstance(hexbeam.field_data['list'], np.ndarray)
    assert np.allclose(hexbeam.field_data['list'], np.arange(n))

    foo = np.arange(n) * 5
    hexbeam.add_field_data(foo, 'foo')
    assert isinstance(hexbeam.field_data['foo'], np.ndarray)
    assert np.allclose(hexbeam.field_data['foo'], foo)

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_scalars('foo')


def test_field_data_string(hexbeam):
    # test `mesh.field_data`
    field_name = 'foo'
    field_value = 'bar'
    hexbeam.field_data[field_name] = field_value
    returned = hexbeam.field_data[field_name]
    assert returned == field_value
    assert isinstance(returned, str)

    # test `mesh.add_field_data`
    field_name = 'eggs'
    field_value = 'ham'
    hexbeam.add_field_data(array=field_value, name=field_name)
    returned = hexbeam.field_data[field_name]
    assert returned == field_value
    assert isinstance(returned, str)

    # test `mesh[name] = data`
    field_name = 'baz'
    field_value = 'a' * hexbeam.n_points
    hexbeam[field_name] = field_value
    returned = hexbeam.field_data[field_name]
    assert returned == field_value
    assert isinstance(returned, str)


@pytest.mark.parametrize('field', [range(5), np.ones((3, 3))[:, 0]])
def test_add_field_data(hexbeam, field):
    hexbeam.add_field_data(field, 'foo')
    assert isinstance(hexbeam.field_data['foo'], np.ndarray)
    assert np.allclose(hexbeam.field_data['foo'], field)


def test_modify_field_data(hexbeam):
    field = range(4)
    hexbeam.add_field_data(range(5), 'foo')
    hexbeam.add_field_data(field, 'foo')
    assert np.allclose(hexbeam.field_data['foo'], field)

    field = range(8)
    hexbeam.field_data['foo'] = field
    assert np.allclose(hexbeam.field_data['foo'], field)


def test_active_scalars_cell(hexbeam):
    hexbeam.add_field_data(range(5), 'foo')
    del hexbeam.point_data['sample_point_scalars']
    del hexbeam.point_data['VTKorigID']
    assert hexbeam.active_scalars_info[1] == 'sample_cell_scalars'


def test_field_data_bad_value(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.field_data['new_array'] = None


def test_copy(hexbeam):
    grid_copy = hexbeam.copy(deep=True)
    grid_copy.points[0] = np.nan
    assert not np.any(np.isnan(hexbeam.points[0]))

    grid_copy_shallow = hexbeam.copy(deep=False)
    grid_copy.points[0] += 0.1
    assert np.all(grid_copy_shallow.points[0] == hexbeam.points[0])


def test_copy_metadata(globe):
    """Ensure metadata is copied correctly."""
    globe.point_data['bitarray'] = np.zeros(globe.n_points, dtype=bool)
    globe.point_data['complex_data'] = np.zeros(globe.n_points, dtype=np.complex128)

    globe_shallow = globe.copy(deep=False)
    assert globe_shallow._active_scalars_info is globe._active_scalars_info
    assert globe_shallow._active_vectors_info is globe._active_vectors_info
    assert globe_shallow._active_tensors_info is globe._active_tensors_info
    assert globe_shallow.point_data['bitarray'].dtype == np.bool_
    assert globe_shallow.point_data['complex_data'].dtype == np.complex128
    assert globe_shallow._association_bitarray_names is globe._association_bitarray_names
    assert globe_shallow._association_complex_names is globe._association_complex_names

    globe_deep = globe.copy(deep=True)
    assert globe_deep._active_scalars_info is not globe._active_scalars_info
    assert globe_deep._active_vectors_info is not globe._active_vectors_info
    assert globe_deep._active_tensors_info is not globe._active_tensors_info
    assert globe_deep._active_scalars_info == globe._active_scalars_info
    assert globe_deep._active_vectors_info == globe._active_vectors_info
    assert globe_deep._active_tensors_info == globe._active_tensors_info
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


def test_set_points():
    dataset = pv.UnstructuredGrid()
    points = np.random.default_rng().random((10, 3))
    dataset.points = pv.vtk_points(points)


def test_make_points_double(hexbeam):
    hexbeam.points = hexbeam.points.astype(np.float32)
    assert hexbeam.points.dtype == np.float32
    hexbeam.points_to_double()
    assert hexbeam.points.dtype == np.double


def test_invalid_points(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.points = None


def test_points_np_bool(hexbeam):
    bool_arr = np.zeros(hexbeam.n_points, np.bool_)
    hexbeam.point_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert hexbeam.point_data['bool_arr'].all()
    assert hexbeam.point_data['bool_arr'].all()
    assert hexbeam.point_data['bool_arr'].dtype == np.bool_


def test_cells_np_bool(hexbeam):
    bool_arr = np.zeros(hexbeam.n_cells, np.bool_)
    hexbeam.cell_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert hexbeam.cell_data['bool_arr'].all()
    assert hexbeam.cell_data['bool_arr'].all()
    assert hexbeam.cell_data['bool_arr'].dtype == np.bool_


def test_field_np_bool(hexbeam):
    bool_arr = np.zeros(hexbeam.n_cells // 3, np.bool_)
    hexbeam.field_data['bool_arr'] = bool_arr
    bool_arr[:] = True
    assert hexbeam.field_data['bool_arr'].all()
    assert hexbeam.field_data['bool_arr'].all()
    assert hexbeam.field_data['bool_arr'].dtype == np.bool_


def test_cells_uint8(hexbeam):
    arr = np.zeros(hexbeam.n_cells, np.uint8)
    hexbeam.cell_data['arr'] = arr
    arr[:] = np.arange(hexbeam.n_cells)
    assert np.allclose(hexbeam.cell_data['arr'], np.arange(hexbeam.n_cells))


def test_points_uint8(hexbeam):
    arr = np.zeros(hexbeam.n_points, np.uint8)
    hexbeam.point_data['arr'] = arr
    arr[:] = np.arange(hexbeam.n_points)
    assert np.allclose(hexbeam.point_data['arr'], np.arange(hexbeam.n_points))


def test_field_uint8(hexbeam):
    n = hexbeam.n_points // 3
    arr = np.zeros(n, np.uint8)
    hexbeam.field_data['arr'] = arr
    arr[:] = np.arange(n)
    assert np.allclose(hexbeam.field_data['arr'], np.arange(n))


def test_bitarray_points(hexbeam):
    n = hexbeam.n_points
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    hexbeam.GetPointData().AddArray(vtk_array)
    assert np.allclose(hexbeam.point_data['bint_arr'], np_array)


def test_bitarray_cells(hexbeam):
    n = hexbeam.n_cells
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    hexbeam.GetCellData().AddArray(vtk_array)
    assert np.allclose(hexbeam.cell_data['bint_arr'], np_array)


def test_bitarray_field(hexbeam):
    n = hexbeam.n_cells // 3
    vtk_array = vtk.vtkBitArray()
    np_array = np.empty(n, np.bool_)
    vtk_array.SetNumberOfTuples(n)
    vtk_array.SetName('bint_arr')
    for i in range(n):
        value = i % 2
        vtk_array.SetValue(i, value)
        np_array[i] = value

    hexbeam.GetFieldData().AddArray(vtk_array)
    assert np.allclose(hexbeam.field_data['bint_arr'], np_array)


def test_html_repr(hexbeam):
    """This just tests to make sure no errors are thrown on the HTML
    representation method for DataSet.
    """
    assert hexbeam._repr_html_() is not None


def test_html_repr_string_scalar(hexbeam):
    array_data = 'data'
    array_name = 'name'
    hexbeam.add_field_data(array_data, array_name)
    assert hexbeam._repr_html_() is not None


@pytest.mark.parametrize('html', [True, False])
@pytest.mark.parametrize('display', [True, False])
def test_print_repr(hexbeam, display, html):
    """This just tests to make sure no errors are thrown on the text friendly
    representation method for DataSet.
    """
    result = hexbeam.head(display=display, html=html)
    assert isinstance(result, str)
    if display and html:
        assert result == ''
    else:
        assert result != ''


def test_invalid_vector(hexbeam):
    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam['vectors'] = np.empty(10)

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam['vectors'] = np.empty((3, 2))

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam['vectors'] = np.empty((3, 3))


def test_no_texture_coordinates(hexbeam):
    assert hexbeam.active_texture_coordinates is None


def test_no_arrows(hexbeam):
    assert hexbeam.arrows is None


def test_arrows():
    sphere = pv.Sphere(radius=math.pi)

    # make cool swirly pattern
    vectors = np.vstack(
        (np.sin(sphere.points[:, 0]), np.cos(sphere.points[:, 1]), np.cos(sphere.points[:, 2])),
    ).T

    # add and scales
    sphere['vectors'] = vectors * 0.3
    sphere.set_active_vectors('vectors')
    assert np.allclose(sphere.active_vectors, vectors * 0.3)
    assert np.allclose(sphere['vectors'], vectors * 0.3)

    assert sphere.active_vectors_info[1] == 'vectors'
    arrows = sphere.arrows
    assert isinstance(arrows, pv.PolyData)
    assert np.any(arrows.points)
    assert arrows.active_vectors_name == 'GlyphVector'


def test_arrows_cell_data():
    box = pv.Box().compute_normals(cell_normals=True, point_normals=False)
    assert box.array_names == ['Normals']
    assert box.cell_data.keys() == ['Normals']
    assert box.arrows is None

    box.set_active_vectors('Normals')
    arrows = box.arrows
    # Test that there are as many arrows as there are vectors
    num_parts_per_arrow = pv.Arrow().split_bodies().n_blocks
    num_parts = arrows.split_bodies().n_blocks
    num_arrows = num_parts / num_parts_per_arrow

    num_vectors = len(box['Normals'])
    assert num_arrows == num_vectors


def test_arrows_ndim_raises(mocker: MockerFixture):
    m = mocker.patch.object(pv.DataSet, 'active_vectors')
    mocker.patch.object(pv.DataSet, 'active_vectors_name')
    m.ndim = 1

    sphere = pv.Sphere(radius=math.pi)
    with pytest.raises(ValueError, match='Active vectors are not vectors.'):
        sphere.arrows  # noqa: B018


def test_set_active_scalars_raises(mocker: MockerFixture):
    sphere = pv.Sphere(radius=math.pi)
    sphere.point_data[(f := 'foo')] = 1

    m = mocker.patch.object(dataset, 'get_array_association')
    m.return_value = 1

    with pytest.raises(
        ValueError,
        match=re.escape('Data field (foo) with type (1) not usable'),
    ):
        sphere.set_active_scalars(f)


def test_set_active_scalars_raises_vtk(mocker: MockerFixture):
    sphere = pv.Sphere(radius=math.pi)
    sphere.point_data[(f := 'foo')] = 1

    m = mocker.patch.object(sphere, 'GetPointData')
    m().SetActiveScalars.return_value = -1

    match = re.escape(
        f'Data field "{f}" with type (FieldAssociation.POINT) could not be set as the '
        f'active scalars'
    )
    with pytest.raises(ValueError, match=match):
        sphere.set_active_scalars(f)


def active_component_consistency_check(grid, component_type, field_association='point'):
    # Tests if the active component (scalars, vectors, tensors) actually reflects
    # the underlying VTK dataset
    component_type = component_type.lower()
    vtk_component_type = component_type.capitalize()

    field_association = field_association.lower()
    vtk_field_association = field_association.capitalize()

    pv_arr = getattr(grid, 'active_' + component_type)
    vtk_arr = getattr(
        getattr(grid, f'Get{vtk_field_association}Data')(),
        f'Get{vtk_component_type}',
    )()

    assert (pv_arr is None and vtk_arr is None) or np.allclose(pv_arr, vtk_to_numpy(vtk_arr))


def test_set_active_vectors(hexbeam):
    vector_arr = np.arange(hexbeam.n_points * 3).reshape([hexbeam.n_points, 3])
    hexbeam.point_data['vector_arr'] = vector_arr
    hexbeam.active_vectors_name = 'vector_arr'
    active_component_consistency_check(hexbeam, 'vectors', 'point')
    assert hexbeam.active_vectors_name == 'vector_arr'
    assert np.allclose(hexbeam.active_vectors, vector_arr)

    hexbeam.active_vectors_name = None
    assert hexbeam.active_vectors_name is None
    active_component_consistency_check(hexbeam, 'vectors', 'point')


def test_set_active_tensors(hexbeam):
    tensor_arr = np.arange(hexbeam.n_points * 9).reshape([hexbeam.n_points, 9])
    hexbeam.point_data['tensor_arr'] = tensor_arr
    hexbeam.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(hexbeam, 'tensors', 'point')
    assert hexbeam.active_tensors_name == 'tensor_arr'
    assert np.allclose(hexbeam.active_tensors, tensor_arr)

    hexbeam.active_tensors_name = None
    assert hexbeam.active_tensors_name is None
    active_component_consistency_check(hexbeam, 'tensors', 'point')


def test_set_texture_coordinates(hexbeam):
    with pytest.raises(TypeError):
        hexbeam.active_texture_coordinates = [1, 2, 3]

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.active_texture_coordinates = np.empty(10)

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.active_texture_coordinates = np.empty((3, 3))

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.active_texture_coordinates = np.empty((hexbeam.n_points, 1))


def test_set_active_vectors_fail(hexbeam):
    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_vectors('not a vector')

    active_component_consistency_check(hexbeam, 'vectors', 'point')
    vector_arr = np.arange(hexbeam.n_points * 3).reshape([hexbeam.n_points, 3])
    hexbeam.point_data['vector_arr'] = vector_arr
    hexbeam.active_vectors_name = 'vector_arr'
    active_component_consistency_check(hexbeam, 'vectors', 'point')

    hexbeam.point_data['scalar_arr'] = np.zeros([hexbeam.n_points])

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_vectors('scalar_arr')

    assert hexbeam.active_vectors_name == 'vector_arr'
    active_component_consistency_check(hexbeam, 'vectors', 'point')


def test_set_active_tensors_fail(hexbeam):
    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_tensors('not a tensor')

    active_component_consistency_check(hexbeam, 'tensors', 'point')
    tensor_arr = np.arange(hexbeam.n_points * 9).reshape([hexbeam.n_points, 9])
    hexbeam.point_data['tensor_arr'] = tensor_arr
    hexbeam.active_tensors_name = 'tensor_arr'
    active_component_consistency_check(hexbeam, 'tensors', 'point')

    hexbeam.point_data['scalar_arr'] = np.zeros([hexbeam.n_points])
    hexbeam.point_data['vector_arr'] = np.zeros([hexbeam.n_points, 3])

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_tensors('scalar_arr')

    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam.set_active_tensors('vector_arr')

    assert hexbeam.active_tensors_name == 'tensor_arr'
    active_component_consistency_check(hexbeam, 'tensors', 'point')


def test_set_active_scalars(hexbeam):
    arr = np.arange(hexbeam.n_cells)
    hexbeam.cell_data['tmp'] = arr
    hexbeam.set_active_scalars('tmp')
    assert np.allclose(hexbeam.active_scalars, arr)
    # Make sure we can set no active scalars
    hexbeam.set_active_scalars(None)
    assert hexbeam.GetPointData().GetScalars() is None
    assert hexbeam.GetCellData().GetScalars() is None


def test_set_active_scalars_name(hexbeam):
    point_keys = list(hexbeam.point_data.keys())
    hexbeam.active_scalars_name = point_keys[0]
    hexbeam.active_scalars_name = None


def test_rename_array_point(hexbeam):
    point_keys = list(hexbeam.point_data.keys())
    old_name = point_keys[0]
    orig_vals = hexbeam[old_name].copy()
    new_name = 'point changed'
    hexbeam.set_active_scalars(old_name, preference='point')
    hexbeam.rename_array(old_name, new_name, preference='point')
    assert new_name in hexbeam.point_data
    assert old_name not in hexbeam.point_data
    assert new_name == hexbeam.active_scalars_name
    assert np.array_equal(orig_vals, hexbeam[new_name])


def test_rename_array_cell(hexbeam):
    cell_keys = list(hexbeam.cell_data.keys())
    old_name = cell_keys[0]
    orig_vals = hexbeam[old_name].copy()
    new_name = 'cell changed'
    hexbeam.rename_array(old_name, new_name)
    assert new_name in hexbeam.cell_data
    assert old_name not in hexbeam.cell_data
    assert np.array_equal(orig_vals, hexbeam[new_name])


def test_rename_array_field(hexbeam):
    hexbeam.field_data['fieldfoo'] = np.array([8, 6, 7])
    field_keys = list(hexbeam.field_data.keys())
    old_name = field_keys[0]
    orig_vals = hexbeam[old_name].copy()
    new_name = 'cell changed'
    hexbeam.rename_array(old_name, new_name)
    assert new_name in hexbeam.field_data
    assert old_name not in hexbeam.field_data
    assert np.array_equal(orig_vals, hexbeam[new_name])


def test_rename_array_raises(mocker: MockerFixture):
    sphere = pv.Sphere(radius=math.pi)

    m = mocker.patch.object(dataset, 'get_array_association')
    m.return_value = None
    f = 'foo'

    with pytest.raises(
        KeyError,
        match=re.escape(f'Array with name {f} not found.'),
    ):
        sphere.rename_array(f, 'bar')


def test_rename_array_doesnt_delete():
    # Regression test for issue #5244
    def make_mesh():
        m = pv.Sphere()
        m.point_data['orig'] = np.ones(m.n_points)
        return m

    mesh = make_mesh()
    was_deleted = [False]

    def on_delete(*_):
        # Would be easier to throw an exception here but even though the exception gets printed to
        # stderr pytest reports the test passing. See #5246 .
        was_deleted[0] = True

    mesh.point_data['orig'].VTKObject.AddObserver('DeleteEvent', on_delete)
    mesh.rename_array('orig', 'renamed')
    assert not was_deleted[0]
    mesh.point_data['renamed'].VTKObject.RemoveAllObservers()
    assert (mesh.point_data['renamed'] == 1).all()


def test_change_name_fail(hexbeam):
    with pytest.raises(KeyError):
        hexbeam.rename_array('not a key', '')


def test_get_cell_array_fail():
    sphere = pv.Sphere()
    with pytest.raises(TypeError):
        sphere.cell_data[None]


def test_get_item(hexbeam):
    with pytest.raises(KeyError):
        hexbeam[0]


def test_set_item(hexbeam):
    with pytest.raises(TypeError):
        hexbeam['tmp'] = None

    # field data
    with pytest.raises(ValueError):  # noqa: PT011
        hexbeam['bad_field'] = range(5)


def test_set_item_range(hexbeam):
    rng = range(hexbeam.n_points)
    hexbeam['pt_rng'] = rng
    assert np.allclose(hexbeam['pt_rng'], rng)


def test_str(hexbeam):
    assert 'UnstructuredGrid' in str(hexbeam)


def test_set_cell_vectors(hexbeam):
    arr = np.random.default_rng().random((hexbeam.n_cells, 3))
    hexbeam.cell_data['_cell_vectors'] = arr
    hexbeam.set_active_vectors('_cell_vectors')
    assert hexbeam.active_vectors_name == '_cell_vectors'
    assert np.allclose(hexbeam.active_vectors, arr)


def test_axis_rotation_invalid():
    with pytest.raises(ValueError):  # noqa: PT011
        pv.axis_rotation(np.empty((3, 3)), 0, inplace=False, axis='not')


def test_axis_rotation_not_inplace():
    p = np.eye(3)
    p_out = pv.axis_rotation(p, 1, inplace=False, axis='x')
    assert not np.allclose(p, p_out)


@pytest.mark.parametrize('name', ['DataSet', 'Grid', 'DataSetFilters', 'PointGrid', 'DataObject'])
def test_init_abstract_class(name):
    klass = getattr(pv, name)
    with pytest.raises(TypeError):
        klass()


def test_string_arrays():
    poly = pv.PolyData(np.random.default_rng().random((10, 3)))
    arr = np.array([f'foo{i}' for i in range(10)])
    poly['foo'] = arr
    back = poly['foo']
    assert len(back) == 10


def test_clear_data():
    # First try on an empty mesh
    grid = pv.ImageData(dimensions=(10, 10, 10))
    # Now try something more complicated
    grid.clear_data()
    grid['foo-p'] = np.random.default_rng().random(grid.n_points)
    grid['foo-c'] = np.random.default_rng().random(grid.n_cells)
    grid.field_data['foo-f'] = np.random.default_rng().random(grid.n_points * grid.n_cells)
    assert grid.n_arrays == 3
    grid.clear_data()
    assert grid.n_arrays == 0


def test_scalars_dict_update():
    mesh = examples.load_uniform()
    n = len(mesh.point_data)
    arrays = {
        'foo': np.arange(mesh.n_points),
        'rand': np.random.default_rng().random(mesh.n_points),
    }
    mesh.point_data.update(arrays)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_data) == n + 2

    # Test update from Table
    table = pv.Table(arrays)
    mesh = examples.load_uniform()
    mesh.point_data.update(table)
    assert 'foo' in mesh.array_names
    assert 'rand' in mesh.array_names
    assert len(mesh.point_data) == n + 2


def test_handle_array_with_null_name():
    poly = pv.PolyData()
    # Add point array with no name
    poly.GetPointData().AddArray(pv.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    pdata = poly.point_data
    assert pdata is not None
    assert len(pdata) == 1
    # Add cell array with no name
    poly.GetCellData().AddArray(pv.convert_array(np.array([])))
    html = poly._repr_html_()
    assert html is not None
    cdata = poly.cell_data
    assert cdata is not None
    assert len(cdata) == 1
    # Add field array with no name
    poly.GetFieldData().AddArray(pv.convert_array(np.array([5, 6])))
    html = poly._repr_html_()
    assert html is not None
    fdata = poly.field_data
    assert fdata is not None
    assert len(fdata) == 1


def test_add_point_array_list(hexbeam):
    rng = range(hexbeam.n_points)
    hexbeam.point_data['tmp'] = rng
    assert np.allclose(hexbeam.point_data['tmp'], rng)


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
    wrapped = pv.PolyData(original, deep=False)
    wrapped.points[:] = 2.8
    orig_points = vtk_to_numpy(original.GetPoints().GetData())
    assert np.allclose(orig_points, wrapped.points)
    # Case 2
    original = vtk.vtkPolyData()
    wrapped = pv.PolyData(original, deep=False)
    wrapped.points = np.random.default_rng().random((5, 3))
    orig_points = vtk_to_numpy(original.GetPoints().GetData())
    assert np.allclose(orig_points, wrapped.points)


def test_find_closest_point():
    sphere = pv.Sphere()
    node = np.array([0, 0.2, 0.2])

    with pytest.raises(TypeError):
        sphere.find_closest_point([1, 2])

    with pytest.raises(ValueError):  # noqa: PT011
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
    mesh = pv.Wavelet()
    node = np.array([0, 0.2, 0.2])
    index = mesh.find_closest_cell(node)
    assert isinstance(index, int)


def test_find_closest_cells():
    mesh = pv.Sphere()
    # simply get the face centers, ordered by cell Id
    fcent = mesh.points[mesh.regular_faces].mean(1)
    fcent_copy = fcent.copy()
    indices = mesh.find_closest_cell(fcent)

    # Make sure we match the face centers
    assert np.allclose(indices, np.arange(mesh.n_faces_strict))

    # Make sure arg was not modified
    assert np.array_equal(fcent, fcent_copy)


def test_find_closest_cell_surface_point():
    mesh = pv.Rectangle()

    point = np.array([0.5, 0.5, -1.0])
    point2 = np.array([1.0, 1.0, -1.0])
    points = np.vstack((point, point2))

    _, closest_point = mesh.find_closest_cell(point, return_closest_point=True)
    assert np.allclose(closest_point, [0.5, 0.5, 0])

    _, closest_points = mesh.find_closest_cell(points, return_closest_point=True)
    assert np.allclose(closest_points, [[0.5, 0.5, 0], [1.0, 1.0, 0]])


def test_find_containing_cell():
    mesh = pv.ImageData(dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0])
    node = np.array([0.3, 0.3, 0.0])
    index = mesh.find_containing_cell(node)
    assert index == 5


def test_find_containing_cells():
    mesh = pv.ImageData(dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0])
    points = np.array([[0.3, 0.3, 0], [0.6, 0.6, 0]])
    points_copy = points.copy()
    indices = mesh.find_containing_cell(points)
    assert np.allclose(indices, [5, 10])
    assert np.array_equal(points, points_copy)


def test_find_cells_along_line():
    mesh = pv.Cube()
    indices = mesh.find_cells_along_line([0, 0, -1], [0, 0, 1])
    assert len(indices) == 2


def test_find_cells_along_line_raises():
    mesh = pv.Cube()
    with pytest.raises(TypeError, match='Point A must be a length three tuple of floats.'):
        mesh.find_cells_along_line([0, 0], [0, 0, 1])

    with pytest.raises(TypeError, match='Point B must be a length three tuple of floats.'):
        mesh.find_cells_along_line([0, 0, -1], [0, 0])


def test_find_cells_intersecting_line():
    mesh = pv.Plane(center=(0.01, 0.5, 1), i_resolution=2, j_resolution=2)
    linea = [0, 0, 0.0]
    lineb = [0.0, 0, 1.0]

    if pv.vtk_version_info >= (9, 2, 0):
        indices = mesh.find_cells_intersecting_line(linea, lineb)
        assert len(indices) == 1

        # test tolerance
        indices = mesh.find_cells_intersecting_line(linea, lineb, tolerance=0.01)
        assert len(indices) == 2

        with pytest.raises(TypeError):
            mesh.find_cells_intersecting_line([0, 0], [1.0, 0, 0.0])

        with pytest.raises(TypeError):
            mesh.find_cells_intersecting_line([0, 0, 0.0], [1.0, 0])

    else:
        with pytest.raises(VTKVersionError):
            indices = mesh.find_cells_intersecting_line(linea, lineb)


def test_find_cells_within_bounds():
    mesh = pv.Cube()

    bounds = [
        mesh.bounds.x_min * 2.0,
        mesh.bounds.x_max * 2.0,
        mesh.bounds.y_min * 2.0,
        mesh.bounds.y_max * 2.0,
        mesh.bounds.z_min * 2.0,
        mesh.bounds.z_max * 2.0,
    ]
    indices = mesh.find_cells_within_bounds(bounds)
    assert len(indices) == mesh.n_cells

    bounds = [
        mesh.bounds.x_min * 0.5,
        mesh.bounds.x_max * 0.5,
        mesh.bounds.y_min * 0.5,
        mesh.bounds.y_max * 0.5,
        mesh.bounds.z_min * 0.5,
        mesh.bounds.z_max * 0.5,
    ]
    indices = mesh.find_cells_within_bounds(bounds)
    assert len(indices) == 0


def test_find_cells_within_bounds_raises():
    mesh = pv.Cube()
    with pytest.raises(
        TypeError,
        match='Bounds must be a length six tuple of floats.',
    ):
        mesh.find_cells_within_bounds([0, 0])


def test_setting_points_by_different_types(hexbeam):
    grid_copy = hexbeam.copy()
    hexbeam.points = grid_copy.points
    assert np.array_equal(hexbeam.points, grid_copy.points)

    hexbeam.points = np.array(grid_copy.points)
    assert np.array_equal(hexbeam.points, grid_copy.points)

    hexbeam.points = grid_copy.points.tolist()
    assert np.array_equal(hexbeam.points, grid_copy.points)

    pgrid = pv.PolyData([0.0, 0.0, 0.0])
    pgrid.points = [1.0, 1.0, 1.0]
    assert np.array_equal(pgrid.points, [[1.0, 1.0, 1.0]])

    pgrid.points = np.array([2.0, 2.0, 2.0])
    assert np.array_equal(pgrid.points, [[2.0, 2.0, 2.0]])


def test_empty_points():
    pdata = pv.PolyData()
    assert np.allclose(pdata.points, np.empty(3))


def test_no_active():
    pdata = pv.PolyData()
    assert pdata.active_scalars is None

    with pytest.raises(TypeError):
        pdata.point_data[None]


def test_get_data_range(hexbeam):
    # Test with blank mesh
    mesh = pv.Sphere()
    mesh.clear_data()
    rng = mesh.get_data_range()
    assert all(np.isnan(rng))
    with pytest.raises(KeyError):
        rng = mesh.get_data_range('some data')

    # Test with some data
    hexbeam.active_scalars_name = 'sample_point_scalars'
    rng = hexbeam.get_data_range()  # active scalars
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = hexbeam.get_data_range('sample_point_scalars', preference='point')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 302))

    rng = hexbeam.get_data_range('sample_cell_scalars', preference='cell')
    assert len(rng) == 2
    assert np.allclose(rng, (1, 40))


def test_actual_memory_size(hexbeam):
    size = hexbeam.actual_memory_size
    assert isinstance(size, int)
    assert size >= 0


def test_copy_structure(hexbeam):
    classname = hexbeam.__class__.__name__
    copy = eval(f'pv.{classname}')()
    copy.copy_structure(hexbeam)
    assert copy.n_cells == hexbeam.n_cells
    assert copy.n_points == hexbeam.n_points
    assert len(copy.field_data) == 0
    assert len(copy.cell_data) == 0
    assert len(copy.point_data) == 0


def test_copy_structure_self(datasets):
    for dataset in datasets:  # noqa: F402
        copied = dataset.copy()
        assert copied is not dataset

        # Copy structure from itself
        copied.copy_structure(copied)
        assert copied.n_points == dataset.n_points
        assert copied.n_cells == dataset.n_cells


def test_copy_attributes(hexbeam):
    classname = hexbeam.__class__.__name__
    copy = eval(f'pv.{classname}')()
    copy.copy_attributes(hexbeam)
    assert copy.n_cells == 0
    assert copy.n_points == 0
    assert copy.field_data.keys() == hexbeam.field_data.keys()
    assert copy.cell_data.keys() == hexbeam.cell_data.keys()
    assert copy.point_data.keys() == hexbeam.point_data.keys()


def test_point_is_inside_cell():
    grid = pv.ImageData(dimensions=(2, 2, 2))
    assert grid.point_is_inside_cell(0, [0.5, 0.5, 0.5])
    assert not grid.point_is_inside_cell(0, [-0.5, -0.5, -0.5])

    assert grid.point_is_inside_cell(0, np.array([0.5, 0.5, 0.5]))

    # cell ind out of range
    with pytest.raises(ValueError):  # noqa: PT011
        grid.point_is_inside_cell(100000, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):  # noqa: PT011
        grid.point_is_inside_cell(-1, [0.5, 0.5, 0.5])

    # cell ind wrong type
    with pytest.raises(TypeError):
        grid.point_is_inside_cell(0.1, [0.5, 0.5, 0.5])

    # point not well formed
    with pytest.raises(TypeError):
        grid.point_is_inside_cell(0, 0.5)
    with pytest.raises(ValueError):  # noqa: PT011
        grid.point_is_inside_cell(0, [0.5, 0.5])

    # multi-dimensional
    in_cell = grid.point_is_inside_cell(0, [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    assert np.array_equal(in_cell, np.array([True, False]))


def test_point_is_inside_cell_raises(mocker: MockerFixture):
    m = mocker.patch.object(pv.ImageData, 'GetCell')
    m().EvaluatePosition.return_value = 2

    grid = pv.ImageData(dimensions=(2, 2, 2))
    with pytest.raises(
        RuntimeError,
        match=re.escape('Computational difficulty encountered for point [0 0 0] in cell 0'),
    ):
        grid.point_is_inside_cell(0, [0, 0, 0])


def test_active_normals(sphere):
    # both cell and point normals
    mesh = sphere.compute_normals()
    assert mesh.active_normals.shape[0] == mesh.n_points

    mesh = sphere.compute_normals(point_normals=False)
    assert mesh.active_normals.shape[0] == mesh.n_cells


@pytest.mark.needs_vtk_version(9, 1, 0, reason='Requires VTK>=9.1.0 for a concrete PointSet class')
def test_cast_to_pointset(sphere):
    sphere = sphere.elevation()
    pointset = sphere.cast_to_pointset()
    assert isinstance(pointset, pv.PointSet)

    assert not np.may_share_memory(sphere.points, pointset.points)
    assert not np.may_share_memory(sphere.active_scalars, pointset.active_scalars)
    assert np.allclose(sphere.points, pointset.points)
    assert np.allclose(sphere.active_scalars, pointset.active_scalars)

    pointset.points[:] = 0
    assert not np.allclose(sphere.points, pointset.points)

    pointset.active_scalars[:] = 0
    assert not np.allclose(sphere.active_scalars, pointset.active_scalars)


@pytest.mark.needs_vtk_version(9, 1, 0, reason='Requires VTK>=9.1.0 for a concrete PointSet class')
def test_cast_to_pointset_implicit(uniform):
    pointset = uniform.cast_to_pointset(pass_cell_data=True)
    assert isinstance(pointset, pv.PointSet)
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
    assert isinstance(points, pv.PolyData)
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
    if pv.vtk_version_info < (9, 1, 0):
        with pytest.raises(VTKVersionError):
            hexbeam.partition(2)
        return
    # split as composite
    n_part = 2
    out = hexbeam.partition(n_part)
    assert isinstance(out, pv.MultiBlock)
    assert len(out) == 2

    # split as unstrucutred grid
    out = hexbeam.partition(hexbeam.n_cells, as_composite=False)
    assert isinstance(hexbeam, pv.UnstructuredGrid)
    assert out.n_points > hexbeam.n_points


def test_explode(datasets):
    for dataset in datasets:  # noqa: F402
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

    # ImageData 3D size 4x4x4
    vol_grid = pv.ImageData(dimensions=(5, 5, 5))
    assert_volume(vol_grid)

    # 2D grid size 4x4
    surf_grid = pv.ImageData(dimensions=(5, 5, 1))
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
    grid = pv.ImageData(dimensions=(5, 5, 5)).extract_surface()
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


def test_raises_cell_neighbors_explicit_structured_grid(datasets_vtk9):
    for dataset in datasets_vtk9:  # noqa: F402
        with pytest.raises(TypeError):
            _ = dataset.cell_neighbors(0)


def test_raises_point_neighbors_ind_overflow(hexbeam):
    with pytest.raises(IndexError):
        _ = hexbeam.point_neighbors(hexbeam.n_points)


def test_raises_cell_neighbors_connections(hexbeam):
    with pytest.raises(ValueError, match='got "topological"'):
        _ = hexbeam.cell_neighbors(0, 'topological')


@pytest.mark.parametrize('grid', grids, ids=ids)
@pytest.mark.parametrize('i0', i0s)
def test_point_cell_ids(grid: DataSet, i0):
    cell_ids = grid.point_cell_ids(i0)

    assert isinstance(cell_ids, list)
    assert all(isinstance(id_, int) for id_ in cell_ids)
    assert all(0 <= id_ < grid.n_cells for id_ in cell_ids)
    assert len(cell_ids) > 0

    # Check that the output cells contain the i0-th point but also that the
    # remaining cells does not contain this point id
    for c in cell_ids:
        assert i0 in grid.get_cell(c).point_ids

    others = [i for i in range(grid.n_cells) if i not in cell_ids]
    for c in others:
        assert i0 not in grid.get_cell(c).point_ids


def test_point_cell_ids_order():
    resolution = 10
    mesh = pv.Sphere(theta_resolution=resolution)
    expected_ids = list(range(resolution))
    actual_ids = mesh.point_cell_ids(0)
    assert actual_ids == expected_ids


@pytest.mark.parametrize('grid', grids_cells, ids=ids_cells)
@pytest.mark.parametrize('i0', i0s)
def test_cell_point_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, 'points')
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all(isinstance(id_, int) for id_ in cell_ids)
    assert all(0 <= id_ < grid.n_cells for id_ in cell_ids)
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


@pytest.mark.parametrize('grid', grids_cells, ids=ids_cells)
@pytest.mark.parametrize('i0', i0s)
def test_cell_edge_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, 'edges')
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all(isinstance(id_, int) for id_ in cell_ids)
    assert all(0 <= id_ < grid.n_cells for id_ in cell_ids)
    assert len(cell_ids) > 0

    # Check that all the neighbors cells share at least one edge with the
    # current cell
    current_points = set()
    current_points.update(frozenset(e.point_ids) for e in cell.edges)

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
@pytest.mark.parametrize('grid', grids_cells[2:], ids=ids_cells[2:])
@pytest.mark.parametrize('i0', i0s)
def test_cell_face_neighbors_ids(grid: DataSet, i0):
    cell_ids = grid.cell_neighbors(i0, 'faces')
    cell = grid.get_cell(i0)

    assert isinstance(cell_ids, list)
    assert all(isinstance(id_, int) for id_ in cell_ids)
    assert all(0 <= id_ < grid.n_cells for id_ in cell_ids)
    assert len(cell_ids) > 0

    # Check that all the neighbors cells share at least one face with the
    # current cell
    current_points = set()
    current_points.update(frozenset(f.point_ids) for f in cell.faces)

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


@pytest.mark.parametrize('grid', grids_cells, ids=ids_cells)
@pytest.mark.parametrize('i0', i0s, ids=lambda x: f'i0={x}')
@pytest.mark.parametrize('n_levels', [1, 3], ids=lambda x: f'n_levels={x}')
@pytest.mark.parametrize(
    'connections',
    ['points', 'edges', 'faces'],
    ids=lambda x: f'connections={x}',
)
def test_cell_neighbors_levels(grid: DataSet, i0, n_levels, connections):
    cell_ids = grid.cell_neighbors_levels(i0, connections=connections, n_levels=n_levels)

    if connections == 'faces' and grid.get_cell(i0).dimension != 3:
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
        assert len(list(cell_ids)) == n_levels
        for ids in cell_ids:
            assert isinstance(ids, list)
            assert all(isinstance(id_, int) for id_ in ids)
            assert all(0 <= id_ < grid.n_cells for id_ in ids)
            assert len(ids) > 0


@pytest.mark.parametrize('grid', grids, ids=ids)
@pytest.mark.parametrize('i0', i0s)
@pytest.mark.parametrize('n_levels', [1, 3])
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
        assert len(list(point_ids)) == n_levels
        for ids in point_ids:
            assert isinstance(ids, list)
            assert all(isinstance(id_, int) for id_ in ids)
            assert all(0 <= id_ < grid.n_points for id_ in ids)
            assert len(ids) > 0


@pytest.fixture
def mesh():
    return examples.load_globe()


def test_active_array_info_deprecated():
    match = 'ActiveArrayInfo is deprecated. Use ActiveArrayInfoTuple instead.'
    with pytest.warns(PyVistaDeprecationWarning, match=match):
        pv.core.dataset.ActiveArrayInfo(association=pv.FieldAssociation.POINT, name='name')
    if pv._version.version_info[:2] > (0, 48):
        msg = 'Remove this deprecated class'
        raise RuntimeError(msg)
