from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
import trimesh

import pyvista as pv
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import MissingDataError
from pyvista.core.utilities.arrays import set_default_active_scalars
from pyvista.core.utilities.points import make_tri_mesh
from pyvista.examples import cells
from tests.core.test_dataobject_filters import grid_with_invalid_arrays  # noqa: F401
from tests.core.test_dataobject_filters import sphere_with_invalid_arrays  # noqa: F401

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_wrap_none():
    # check against the "None" edge case
    assert pv.wrap(None) is None


def test_wrap_pyvista_ndarray(sphere):
    pd = pv.wrap(sphere.points)
    assert isinstance(pd, pv.PolyData)


def test_wrap_raises():
    with pytest.raises(
        NotImplementedError,
        match=r'NumPy array could not be wrapped pyvista.',
    ):
        pv.wrap(np.zeros((42, 42, 42, 42)))


@pytest.fixture
def meshio_grid_with_invalid_arrays(grid_with_invalid_arrays):  # noqa: F811
    grid = grid_with_invalid_arrays
    assert len(grid['foo']) == 10
    # Meshio does internal checks so we can't convert directly:
    match = 'len(points) = 99, but len(point_data["foo"]) = 10'
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.to_meshio(grid)
    # Clear data and add it back after init
    point_data = grid.point_data.items()
    grid.clear_data()
    meshio_grid = pv.to_meshio(grid)
    meshio_grid.point_data.update(point_data)
    assert len(meshio_grid.point_data['foo']) == 10
    return meshio_grid


@pytest.fixture
def trimesh_mesh_with_invalid_arrays(sphere_with_invalid_arrays):  # noqa: F811
    poly = sphere_with_invalid_arrays
    assert len(poly['foo']) == 10
    # Trimesh doesn't do internal checks so we can convert without error
    trimesh_poly = pv.to_trimesh(poly)
    assert len(trimesh_poly.vertex_attributes['foo']) == 10
    return trimesh_poly


def test_wrap_invalid_vtk_mesh_warns(sphere_with_invalid_arrays):  # noqa: F811
    vtk_poly = _vtk.vtkPolyData()
    vtk_poly.ShallowCopy(sphere_with_invalid_arrays)
    with pytest.warns(pv.InvalidMeshWarning, match='Invalid array'):
        pv.wrap(vtk_poly)


def test_wrap_invalid_trimesh_raises(trimesh_mesh_with_invalid_arrays):
    with pytest.raises(ValueError, match='Invalid array'):
        pv.wrap(trimesh_mesh_with_invalid_arrays)


def test_wrap_invalid_meshio_raises(meshio_grid_with_invalid_arrays):
    with pytest.raises(ValueError, match='Invalid array'):
        pv.wrap(meshio_grid_with_invalid_arrays)


def test_wrap_vtk_not_supported_raises(mocker: MockerFixture):
    m = mocker.patch.object(pv, '_wrappers')
    m.__getitem__.side_effect = KeyError
    m2 = mocker.MagicMock()
    m2.GetClassName.return_value = (f := 'foo')

    with pytest.raises(
        TypeError,
        match=re.escape(f'VTK data type ({f}) is not currently supported by pyvista.'),
    ):
        pv.wrap(m2)


def test_wrap_raises_unable():
    with pytest.raises(
        NotImplementedError, match=re.escape("Unable to wrap (<class 'int'>) into a pyvista type.")
    ):
        pv.wrap(1)


# NOTE: It's not necessary to test all data types here, several of the
# most used ones.  We're just checking that we can wrap VTK data types.
@pytest.mark.parametrize(
    'dtypes',
    [
        (np.float64, _vtk.vtkDoubleArray),
        (np.float32, _vtk.vtkFloatArray),
        (np.int64, _vtk.vtkTypeInt64Array),
        (np.int32, _vtk.vtkTypeInt32Array),
        (np.int8, _vtk.vtkSignedCharArray),
        (np.uint8, _vtk.vtkUnsignedCharArray),
    ],
)
def test_wrap_pyvista_ndarray_vtk(dtypes):
    np_dtype, vtk_class = dtypes
    np_array = np.array([[0, 10, 20], [-10, -200, 0], [0.5, 0.667, 0]]).astype(np_dtype)

    vtk_array = vtk_class()
    vtk_array.SetNumberOfComponents(3)
    vtk_array.SetNumberOfValues(9)
    for i in range(9):
        vtk_array.SetValue(i, np_array.flat[i])

    wrapped = pv.wrap(vtk_array)
    assert np.allclose(wrapped, np_array)
    assert wrapped.dtype == np_array.dtype


def test_wrap_trimesh():
    points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    faces = [[0, 1, 2]]
    tmesh = trimesh.Trimesh(points, faces=faces, process=False)
    mesh = pv.wrap(tmesh)
    assert isinstance(mesh, pv.PolyData)

    assert np.allclose(tmesh.vertices, mesh.points)
    assert np.allclose(tmesh.faces, mesh.faces[1:])

    assert mesh.active_texture_coordinates is None

    uvs = [[0, 0], [0, 1], [1, 0]]
    tmesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh_with_uv = pv.wrap(tmesh)

    assert mesh_with_uv.active_texture_coordinates is not None
    assert np.allclose(mesh_with_uv.active_texture_coordinates, uvs)


def test_to_trimesh_triangulate():
    match = (
        'Mesh must be all triangles to convert to Trimesh object.\n'
        'Use `triangulate=True` to automatically convert to a triangle surface mesh.'
    )

    uniform = pv.Wavelet()
    assert isinstance(uniform, pv.ImageData)
    with pytest.raises(pv.NotAllTrianglesError, match=match):
        pv.to_trimesh(uniform)
    out = pv.to_trimesh(uniform, triangulate=True)
    assert isinstance(out, trimesh.Trimesh)

    quad_poly = cells.Quadrilateral().extract_geometry()
    assert isinstance(quad_poly, pv.PolyData)
    with pytest.raises(pv.NotAllTrianglesError, match=match):
        pv.to_trimesh(quad_poly)
    out = pv.to_trimesh(quad_poly, triangulate=True)
    assert isinstance(out, trimesh.Trimesh)

    grid_tetra = cells.Tetrahedron()
    assert isinstance(grid_tetra, pv.UnstructuredGrid)
    with pytest.raises(pv.NotAllTrianglesError, match=match):
        pv.to_trimesh(grid_tetra)
    out = pv.to_trimesh(grid_tetra, triangulate=True)
    assert isinstance(out, trimesh.Trimesh)

    poly_tetra = grid_tetra.extract_geometry()
    assert isinstance(poly_tetra, pv.PolyData)
    out = pv.to_trimesh(poly_tetra)
    assert isinstance(out, trimesh.Trimesh)

    grid_tri = cells.Triangle()
    assert isinstance(grid_tri, pv.UnstructuredGrid)
    pv.to_trimesh(grid_tri)
    assert isinstance(out, trimesh.Trimesh)

    poly_tri = grid_tri.extract_geometry()
    assert isinstance(poly_tri, pv.PolyData)
    pv.to_trimesh(poly_tri)
    assert isinstance(out, trimesh.Trimesh)


def test_to_trimesh_raises(sphere):
    match = "pass_data must be one of: \n\t[True, False, 'point', 'cell', 'field']"
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.to_trimesh(sphere, pass_data='foo')

    tmesh = pv.to_trimesh(sphere)
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.from_trimesh(tmesh, pass_data=dict)

    match = (
        "mesh must be an instance of <class 'pyvista.core.dataset.DataSet'>. "
        "Got <class 'trimesh.base.Trimesh'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.to_trimesh(tmesh)

    match = (
        "mesh must be an instance of <class 'trimesh.base.Trimesh'>. "
        "Got <class 'pyvista.core.pointset.PolyData'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.from_trimesh(sphere)


def test_to_from_trimesh_empty_mesh():
    mesh = pv.ImageData()
    tmesh = pv.to_trimesh(mesh)
    assert isinstance(tmesh, trimesh.Trimesh)

    pvmesh = pv.from_trimesh(tmesh)
    assert isinstance(pvmesh, pv.PolyData)


def test_to_from_trimesh_points_faces(ant):
    ant.points_to_double()
    tmesh = pv.to_trimesh(ant)
    assert np.shares_memory(tmesh.vertices, ant.points)
    assert np.shares_memory(tmesh.faces, ant.regular_faces)

    pvmesh = pv.from_trimesh(tmesh)
    assert np.shares_memory(pvmesh.points, ant.points)
    assert np.shares_memory(pvmesh.regular_faces, ant.regular_faces)


def test_to_from_trimesh_texture_coordinates(ant):
    texture_coordinates_name = 'uv coordinates'
    texture_coordinates_array = np.random.default_rng().random((ant.n_points, 2))
    ant.point_data[texture_coordinates_name] = texture_coordinates_array
    ant.point_data.active_texture_coordinates_name = texture_coordinates_name

    tmesh = pv.to_trimesh(ant, pass_data=False)
    assert not isinstance(tmesh.visual, trimesh.visual.TextureVisuals)

    tmesh = pv.to_trimesh(ant, pass_data='cell')
    assert not isinstance(tmesh.visual, trimesh.visual.TextureVisuals)

    tmesh = pv.to_trimesh(ant)

    actual_array = tmesh.visual.uv
    assert np.allclose(actual_array, texture_coordinates_array)
    assert np.shares_memory(actual_array, texture_coordinates_array)
    assert texture_coordinates_name not in tmesh.vertex_attributes

    # Test we have the same array round-trip
    pvmesh = pv.from_trimesh(tmesh)
    actual_array = pvmesh.active_texture_coordinates
    assert np.allclose(actual_array, texture_coordinates_array)
    assert np.shares_memory(actual_array, texture_coordinates_array)
    assert 'uv' not in pvmesh.point_data

    pvmesh = pv.from_trimesh(tmesh, pass_data=False)
    assert pvmesh.active_texture_coordinates is None

    pvmesh = pv.from_trimesh(tmesh, pass_data='cell')
    assert pvmesh.active_texture_coordinates is None


def test_to_from_trimesh_point_normals(ant):
    point_normals_name = 'point normals'
    point_normals_array = np.random.default_rng().random((ant.n_points, 3))
    ant.point_data[point_normals_name] = point_normals_array
    ant.point_data.active_normals_name = point_normals_name

    tmesh = pv.to_trimesh(ant)

    actual_array = tmesh.vertex_normals
    assert np.allclose(actual_array, point_normals_array)
    assert np.shares_memory(actual_array, point_normals_array)
    assert point_normals_name not in tmesh.vertex_attributes

    # Trimesh uses a cache or re-computes normals on-the-fly, so we don't store them when
    # converting back
    pvmesh = pv.from_trimesh(tmesh)
    assert pvmesh.array_names == []


def test_to_from_trimesh_cell_normals(ant):
    cell_normals_name = 'cells normals'
    cell_normals_array = ant.copy().compute_normals().cell_data['Normals'].astype(float)
    ant.cell_data[cell_normals_name] = cell_normals_array
    ant.cell_data.active_normals_name = cell_normals_name

    tmesh = pv.to_trimesh(ant)

    actual_array = tmesh.face_normals
    assert np.allclose(actual_array, cell_normals_array)
    assert np.shares_memory(actual_array, cell_normals_array)
    assert cell_normals_name not in tmesh.face_attributes

    # Trimesh uses a cache or re-computes normals on-the-fly, so we don't store them when
    # converting back
    pvmesh = pv.from_trimesh(tmesh)
    assert pvmesh.array_names == []


def test_to_from_trimesh_point_data(ant):
    point_data_name = 'point data'
    point_data_array = np.arange(ant.n_points)
    ant.point_data[point_data_name] = point_data_array

    tmesh = pv.to_trimesh(ant, pass_data=False)
    assert tmesh.vertex_attributes == {}

    tmesh = pv.to_trimesh(ant, pass_data='cell')
    assert tmesh.vertex_attributes == {}

    tmesh = pv.to_trimesh(ant)

    actual_attributes = tmesh.vertex_attributes
    assert ant.point_data.keys() == list(actual_attributes.keys())
    actual_array = actual_attributes[point_data_name]
    assert np.allclose(actual_array, point_data_array)
    assert np.shares_memory(actual_array, point_data_array)

    # Test we have the same array round-trip
    for pass_data in [True, 'point', ['point']]:
        pvmesh = pv.from_trimesh(tmesh, pass_data=pass_data)
        actual_array = pvmesh.point_data[point_data_name]
        assert np.allclose(actual_array, point_data_array)
        assert np.shares_memory(actual_array, point_data_array)

    pvmesh = pv.from_trimesh(tmesh, pass_data=False)
    assert pvmesh.point_data.items() == []

    pvmesh = pv.from_trimesh(tmesh, pass_data='cell')
    assert pvmesh.point_data.items() == []


def test_to_from_trimesh_cell_data(ant):
    cell_data_name = 'cell data'
    cell_data_array = np.arange(ant.n_cells)
    ant.cell_data[cell_data_name] = cell_data_array

    tmesh = pv.to_trimesh(ant, pass_data=False)
    assert tmesh.face_attributes == {}

    tmesh = pv.to_trimesh(ant, pass_data='point')
    assert tmesh.face_attributes == {}

    tmesh = pv.to_trimesh(ant)

    actual_attributes = tmesh.face_attributes
    assert ant.cell_data.keys() == list(actual_attributes.keys())
    actual_array = actual_attributes[cell_data_name]
    assert np.allclose(actual_array, cell_data_array)
    assert np.shares_memory(actual_array, cell_data_array)

    # Test we have the same array round-trip
    for pass_data in [True, 'cell', ['cell']]:
        pvmesh = pv.from_trimesh(tmesh, pass_data=pass_data)
        actual_array = pvmesh.cell_data[cell_data_name]
        assert np.allclose(actual_array, cell_data_array)
        assert np.shares_memory(actual_array, cell_data_array)

    pvmesh = pv.from_trimesh(tmesh, pass_data=False)
    assert pvmesh.cell_data.items() == []

    pvmesh = pv.from_trimesh(tmesh, pass_data='point')
    assert pvmesh.cell_data.items() == []


def test_to_from_trimesh_field_data(ant):
    field_data_name = 'field data'
    field_data_array = np.array([42, 67])
    ant.field_data[field_data_name] = field_data_array

    tmesh = pv.to_trimesh(ant, pass_data=False)
    assert tmesh.metadata == {}

    tmesh = pv.to_trimesh(ant, pass_data=['point', 'cell'])
    assert tmesh.metadata == {}

    tmesh = pv.to_trimesh(ant)

    metadata = tmesh.metadata
    assert ant.field_data.keys() == list(metadata.keys())
    actual_array = metadata[field_data_name]
    assert np.array_equal(actual_array, field_data_array)
    assert np.shares_memory(actual_array, field_data_array)

    # Test we have the same array round-trip
    for pass_data in [True, 'field', ['field']]:
        pvmesh = pv.from_trimesh(tmesh, pass_data=pass_data)
        actual_array = pvmesh.field_data[field_data_name]
        assert np.allclose(actual_array, field_data_array)
        assert np.shares_memory(actual_array, field_data_array)

    pvmesh = pv.from_trimesh(tmesh, pass_data=False)
    assert pvmesh.field_data.items() == []

    pvmesh = pv.from_trimesh(tmesh, pass_data=['point', 'cell'])
    assert pvmesh.field_data.items() == []


def test_to_from_trimesh_user_dict(ant):
    user_dict = {'ham': 'eggs'}
    ant.user_dict = user_dict

    tmesh = pv.to_trimesh(ant, pass_data=False)
    assert tmesh.metadata == {}

    tmesh = pv.to_trimesh(ant, pass_data=['point', 'cell'])
    assert tmesh.metadata == {}

    tmesh = pv.to_trimesh(ant)

    metadata = tmesh.metadata
    assert metadata == user_dict

    # Test we have the same array round-trip
    pvmesh = pv.from_trimesh(tmesh)
    round_trip_dict = pvmesh.user_dict
    assert round_trip_dict == user_dict

    pvmesh = pv.from_trimesh(tmesh, pass_data=False)
    assert pvmesh.field_data.items() == []

    pvmesh = pv.from_trimesh(tmesh, pass_data=['point', 'cell'])
    assert pvmesh.field_data.items() == []

    tmesh.metadata['bad_value'] = object
    match = (
        "Unable to store metadata key 'bad_value' with value type <class 'type'>.\n"
        'Only NumPy arrays or JSON-serializable values are supported.'
    )
    with pytest.warns(UserWarning, match=match):
        pv.from_trimesh(tmesh)


def test_make_tri_mesh(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        make_tri_mesh(sphere.points, sphere.faces)

    with pytest.raises(ValueError):  # noqa: PT011
        make_tri_mesh(sphere.points[:, :1], sphere.faces)

    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    mesh = make_tri_mesh(sphere.points, faces)

    assert np.allclose(sphere.points, mesh.points)
    assert np.allclose(sphere.faces, mesh.faces)


def test_wrappers():
    vtk_data = _vtk.vtkPolyData()
    pv_data = pv.wrap(vtk_data)
    assert isinstance(pv_data, pv.PolyData)

    class Foo(pv.PolyData):
        """A user defined subclass of pv.PolyData."""

    default_wrappers = pv._wrappers.copy()
    # Use try...finally to set and reset _wrappers
    try:
        pv._wrappers['vtkPolyData'] = Foo

        pv_data = pv.wrap(vtk_data)
        assert isinstance(pv_data, Foo)

        tri_data = pv_data.delaunay_2d()

        assert isinstance(tri_data, Foo)

        image = pv.ImageData()
        surface = image.extract_surface()

        assert isinstance(surface, Foo)

        surface.delaunay_2d(inplace=True)
        assert isinstance(surface, Foo)

        sphere = pv.Sphere()
        assert isinstance(sphere, Foo)

        circle = pv.Circle()
        assert isinstance(circle, Foo)

    finally:
        pv._wrappers = default_wrappers  # always reset back to default


def test_wrap_no_copy():
    # makes sure that wrapping an already wrapped object returns source
    mesh = pv.Wavelet()
    wrapped = pv.wrap(mesh)
    assert mesh == wrapped
    assert wrapped is mesh

    mesh = _vtk.vtkPolyData()
    wrapped = pv.wrap(mesh)
    assert wrapped == pv.wrap(wrapped)
    assert wrapped is pv.wrap(wrapped)


def test_inheritance_no_wrappers():
    class Foo(pv.PolyData):
        pass

    # inplace operations do not change type
    mesh = Foo(pv.Sphere())
    mesh.decimate(0.5, inplace=True)
    assert isinstance(mesh, Foo)

    # without using _wrappers, we need to explicitly handle inheritance
    mesh = Foo(pv.Sphere())
    new_mesh = mesh.decimate(0.5)
    assert isinstance(new_mesh, pv.PolyData)
    foo_new_mesh = Foo(new_mesh)
    assert isinstance(foo_new_mesh, Foo)


def test_array_association():
    # TODO: cover vtkTable/ROW association case
    mesh = pv.PolyData()
    FieldAssociation = pv.FieldAssociation

    # single match cases
    mesh.point_data['p'] = []
    mesh.cell_data['c'] = []
    mesh.field_data['f'] = ['foo']
    for preference in 'point', 'cell', 'field':
        assoc = mesh.get_array_association('p', preference=preference)
        assert assoc == FieldAssociation.POINT
        assoc = mesh.get_array_association('c', preference=preference)
        assert assoc == FieldAssociation.CELL
        assoc = mesh.get_array_association('f', preference=preference)
        assert assoc == FieldAssociation.NONE

    # multiple match case
    mesh.point_data['common'] = []
    mesh.cell_data['common'] = []
    mesh.field_data['common'] = ['foo']
    assoc = mesh.get_array_association('common', preference='point')
    assert assoc == FieldAssociation.POINT
    assoc = mesh.get_array_association('common', preference='cell')
    assert assoc == FieldAssociation.CELL
    assoc = mesh.get_array_association('common', preference='field')
    assert assoc == FieldAssociation.NONE

    # regression test against overly suggestive preference
    mesh.clear_cell_data()  # point and field left
    assoc = mesh.get_array_association('common', 'cell')
    assert assoc != FieldAssociation.CELL

    # missing cases
    mesh.clear_data()
    with pytest.raises(KeyError, match=r'not present in this dataset.'):
        assoc = mesh.get_array_association('missing')
    assoc = pv.get_array_association(mesh, 'missing', err=False)
    assert assoc == FieldAssociation.NONE

    with pytest.raises(ValueError, match=r'not supported.'):
        mesh.get_array_association('name', preference='row')


def test_set_default_active_vectors():
    mesh = pv.Sphere()
    mesh.clear_data()  # make sure we have a clean mesh with no arrays to start

    assert mesh.active_vectors_name is None

    # Point data vectors
    mesh['vec_point'] = np.ones((mesh.n_points, 3))
    pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name == 'vec_point'
    mesh.clear_data()

    # Cell data vectors
    mesh['vec_cell'] = np.ones((mesh.n_cells, 3))
    pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name == 'vec_cell'
    mesh.clear_data()

    # Raises if no data is present
    with pytest.raises(MissingDataError):
        pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None

    # Raises if no vector-like data is present
    mesh['scalar_data'] = np.ones((mesh.n_points, 1))
    with pytest.raises(MissingDataError):
        pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None
    mesh.clear_data()

    # Raises if multiple vector-like data is present
    mesh['vec_data1'] = np.ones((mesh.n_points, 3))
    mesh['vec_data2'] = np.ones((mesh.n_points, 3))
    with pytest.raises(AmbiguousDataError):
        pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None
    mesh.clear_data()

    # Raises if multiple vector-like data in cell and point
    mesh['vec_data1'] = np.ones((mesh.n_points, 3))
    mesh['vec_data2'] = np.ones((mesh.n_cells, 3))
    with pytest.raises(AmbiguousDataError):
        pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None

    # Raises if multiple vector-like data with same name
    mesh['vec_data'] = np.ones((mesh.n_points, 3))
    mesh['vec_data'] = np.ones((mesh.n_cells, 3))
    with pytest.raises(AmbiguousDataError):
        pv.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None


def test_set_default_active_scalarrs():
    mesh = pv.Sphere()
    mesh.clear_data()  # make sure we have a clean mesh with no arrays to start

    assert mesh.active_scalars_name is None

    # Point data scalars
    mesh['scalar_point'] = np.ones(mesh.n_points)
    mesh.set_active_scalars(None)
    set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == 'scalar_point'
    mesh.clear_data()

    # Cell data scalars
    mesh['scalar_cell'] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == 'scalar_cell'
    mesh.clear_data()

    # Point data scalars multidimensional
    mesh['scalar_point'] = np.ones((mesh.n_points, 3))
    mesh.set_active_scalars(None)
    set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == 'scalar_point'
    mesh.clear_data()

    # Cell data scalars multidimensional
    mesh['scalar_cell'] = np.ones((mesh.n_cells, 3))
    mesh.set_active_scalars(None)
    set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == 'scalar_cell'
    mesh.clear_data()

    # Raises if no data is present
    with pytest.raises(MissingDataError):
        set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None

    # Raises if multiple scalar-like data is present
    mesh['scalar_data1'] = np.ones(mesh.n_points)
    mesh['scalar_data2'] = np.ones(mesh.n_points)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None
    mesh.clear_data()

    # Raises if multiple scalar-like data in cell and point
    mesh['scalar_data1'] = np.ones(mesh.n_points)
    mesh['scalar_data2'] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None

    # Raises if multiple scalar-like data with same name
    mesh['scalar_data'] = np.ones(mesh.n_points)
    mesh['scalar_data'] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None


def test_vtk_points_deep_shallow():
    points = np.array([[0.0, 0.0, 0.0]])
    vtk_points = pv.vtk_points(points, deep=False)

    assert vtk_points.GetNumberOfPoints() == 1
    assert np.array_equal(vtk_points.GetPoint(0), points[0])

    # test shallow copy
    vtk_points.SetPoint(0, [1.0, 1.0, 1.0])

    assert np.array_equal(vtk_points.GetPoint(0), points[0])
    assert np.array_equal(vtk_points.GetPoint(0), [1.0, 1.0, 1.0])

    # test deep copy

    points = np.array([[0.0, 0.0, 0.0]])
    vtk_points = pv.vtk_points(points, deep=True)

    vtk_points.SetPoint(0, [1.0, 1.0, 1.0])

    assert not np.array_equal(vtk_points.GetPoint(0), points[0])
    assert np.array_equal(points[0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ('force_float', 'expected_data_type'),
    [(False, np.int64), (True, np.float32)],
)
def test_vtk_points_force_float(force_float, expected_data_type):
    np_points = np.array([[1, 2, 3]], dtype=np.int64)
    if force_float:
        with pytest.warns(UserWarning, match='Points is not a float type'):
            vtk_points = pv.vtk_points(np_points, force_float=force_float)
    else:
        vtk_points = pv.vtk_points(np_points, force_float=force_float)
    as_numpy = _vtk.vtk_to_numpy(vtk_points.GetData())

    assert as_numpy.dtype == expected_data_type


def test_vtk_points_allow_empty():
    pv.vtk_points([], allow_empty=True)
    match = 'points has shape (0,) which is not allowed. Shape must be one of [3, (-1, 3)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.vtk_points([], allow_empty=False)
