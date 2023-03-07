import os

from PIL import Image
import numpy as np
import pytest
import trimesh
import vtk
from vtk.util import numpy_support

import pyvista
from pyvista import _vtk
from pyvista.errors import AmbiguousDataError, MissingDataError


def test_wrap_none():
    # check against the "None" edge case
    assert pyvista.wrap(None) is None


def test_wrap_pyvista_ndarray(sphere):
    pd = pyvista.wrap(sphere.points)
    assert isinstance(pd, pyvista.PolyData)


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
    np_array = np.array([[0, 10, 20], [-10, -200, 0], [0.5, 0.667, 0]], dtype=np_dtype)

    vtk_array = vtk_class()
    vtk_array.SetNumberOfComponents(3)
    vtk_array.SetNumberOfValues(9)
    for i in range(9):
        vtk_array.SetValue(i, np_array.flat[i])

    wrapped = pyvista.wrap(vtk_array)
    assert np.allclose(wrapped, np_array)
    assert wrapped.dtype == np_array.dtype


def test_wrap_trimesh():
    points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    faces = [[0, 1, 2]]
    tmesh = trimesh.Trimesh(points, faces=faces, process=False)
    mesh = pyvista.wrap(tmesh)
    assert isinstance(mesh, pyvista.PolyData)

    assert np.allclose(tmesh.vertices, mesh.points)
    assert np.allclose(tmesh.faces, mesh.faces[1:])

    assert mesh.active_t_coords is None

    uvs = [[0, 0], [0, 1], [1, 0]]
    tmesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh_with_uv = pyvista.wrap(tmesh)

    assert mesh_with_uv.active_t_coords is not None
    assert np.allclose(mesh_with_uv.active_t_coords, uvs)


def test_make_tri_mesh(sphere):
    with pytest.raises(ValueError):
        pyvista.make_tri_mesh(sphere.points, sphere.faces)

    with pytest.raises(ValueError):
        pyvista.make_tri_mesh(sphere.points[:, :1], sphere.faces)

    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    mesh = pyvista.make_tri_mesh(sphere.points, faces)

    assert np.allclose(sphere.points, mesh.points)
    assert np.allclose(sphere.faces, mesh.faces)


def test_wrappers():
    vtk_data = vtk.vtkPolyData()
    pv_data = pyvista.wrap(vtk_data)
    assert isinstance(pv_data, pyvista.PolyData)

    class Foo(pyvista.PolyData):
        """A user defined subclass of pyvista.PolyData."""

        pass

    default_wrappers = pyvista._wrappers.copy()
    # Use try...finally to set and reset _wrappers
    try:
        pyvista._wrappers['vtkPolyData'] = Foo

        pv_data = pyvista.wrap(vtk_data)
        assert isinstance(pv_data, Foo)

        tri_data = pv_data.delaunay_2d()

        assert isinstance(tri_data, Foo)

        uniform_grid = pyvista.UniformGrid()
        surface = uniform_grid.extract_surface()

        assert isinstance(surface, Foo)

        surface.delaunay_2d(inplace=True)
        assert isinstance(surface, Foo)

        sphere = pyvista.Sphere()
        assert isinstance(sphere, Foo)

        circle = pyvista.Circle()
        assert isinstance(circle, Foo)

    finally:
        pyvista._wrappers = default_wrappers  # always reset back to default


def test_wrap_no_copy():
    # makes sure that wrapping an already wrapped object returns source
    mesh = pyvista.Wavelet()
    wrapped = pyvista.wrap(mesh)
    assert mesh == wrapped
    assert wrapped is mesh

    mesh = vtk.vtkPolyData()
    wrapped = pyvista.wrap(mesh)
    assert wrapped == pyvista.wrap(wrapped)
    assert wrapped is pyvista.wrap(wrapped)


def test_inheritance_no_wrappers():
    class Foo(pyvista.PolyData):
        pass

    # inplace operations do not change type
    mesh = Foo(pyvista.Sphere())
    mesh.decimate(0.5, inplace=True)
    assert isinstance(mesh, Foo)

    # without using _wrappers, we need to explicitly handle inheritance
    mesh = Foo(pyvista.Sphere())
    new_mesh = mesh.decimate(0.5)
    assert isinstance(new_mesh, pyvista.PolyData)
    foo_new_mesh = Foo(new_mesh)
    assert isinstance(foo_new_mesh, Foo)


def test_skybox(tmpdir):
    path = str(tmpdir.mkdir("tmpdir"))
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    filenames = []
    for suffix in sets:
        image = Image.new('RGB', (10, 10))
        filename = os.path.join(path, suffix + '.jpg')
        image.save(filename)
        filenames.append(filename)

    skybox = pyvista.cubemap(path)
    assert isinstance(skybox, pyvista.Texture)

    with pytest.raises(FileNotFoundError, match='Unable to locate'):
        pyvista.cubemap('')

    skybox = pyvista.cubemap_from_filenames(filenames)
    assert isinstance(skybox, pyvista.Texture)

    with pytest.raises(ValueError, match='must contain 6 paths'):
        pyvista.cubemap_from_filenames(image_paths=['/path'])


def test_numpy_to_texture():
    tex_im = np.ones((1024, 1024, 3), dtype=np.float64) * 255
    with pytest.warns(UserWarning, match='np.uint8'):
        tex = pyvista.numpy_to_texture(tex_im)
    assert isinstance(tex, pyvista.Texture)
    assert tex.to_array().dtype == np.uint8


def test_array_association():
    # TODO: cover vtkTable/ROW association case
    mesh = pyvista.PolyData()
    FieldAssociation = pyvista.FieldAssociation

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
    with pytest.raises(KeyError, match='not present in this dataset.'):
        assoc = mesh.get_array_association('missing')
    assoc = pyvista.get_array_association(mesh, 'missing', err=False)
    assert assoc == FieldAssociation.NONE

    with pytest.raises(ValueError, match='not supported.'):
        mesh.get_array_association('name', preference='row')


def test_set_default_active_vectors():
    mesh = pyvista.Sphere()
    mesh.clear_data()  # make sure we have a clean mesh with no arrays to start

    assert mesh.active_vectors_name is None

    # Point data vectors
    mesh["vec_point"] = np.ones((mesh.n_points, 3))
    pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name == "vec_point"
    mesh.clear_data()

    # Cell data vectors
    mesh["vec_cell"] = np.ones((mesh.n_cells, 3))
    pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name == "vec_cell"
    mesh.clear_data()

    # Raises if no data is present
    with pytest.raises(MissingDataError):
        pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None

    # Raises if no vector-like data is present
    mesh["scalar_data"] = np.ones((mesh.n_points, 1))
    with pytest.raises(MissingDataError):
        pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None
    mesh.clear_data()

    # Raises if multiple vector-like data is present
    mesh["vec_data1"] = np.ones((mesh.n_points, 3))
    mesh["vec_data2"] = np.ones((mesh.n_points, 3))
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None
    mesh.clear_data()

    # Raises if multiple vector-like data in cell and point
    mesh["vec_data1"] = np.ones((mesh.n_points, 3))
    mesh["vec_data2"] = np.ones((mesh.n_cells, 3))
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None

    # Raises if multiple vector-like data with same name
    mesh["vec_data"] = np.ones((mesh.n_points, 3))
    mesh["vec_data"] = np.ones((mesh.n_cells, 3))
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_vectors(mesh)
    assert mesh.active_vectors_name is None


def test_set_default_active_scalarrs():
    mesh = pyvista.Sphere()
    mesh.clear_data()  # make sure we have a clean mesh with no arrays to start

    assert mesh.active_scalars_name is None

    # Point data scalars
    mesh["scalar_point"] = np.ones(mesh.n_points)
    mesh.set_active_scalars(None)
    pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == "scalar_point"
    mesh.clear_data()

    # Cell data scalars
    mesh["scalar_cell"] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == "scalar_cell"
    mesh.clear_data()

    # Point data scalars multidimensional
    mesh["scalar_point"] = np.ones((mesh.n_points, 3))
    mesh.set_active_scalars(None)
    pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == "scalar_point"
    mesh.clear_data()

    # Cell data scalars multidimensional
    mesh["scalar_cell"] = np.ones((mesh.n_cells, 3))
    mesh.set_active_scalars(None)
    pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name == "scalar_cell"
    mesh.clear_data()

    # Raises if no data is present
    with pytest.raises(MissingDataError):
        pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None

    # Raises if multiple scalar-like data is present
    mesh["scalar_data1"] = np.ones(mesh.n_points)
    mesh["scalar_data2"] = np.ones(mesh.n_points)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None
    mesh.clear_data()

    # Raises if multiple scalar-like data in cell and point
    mesh["scalar_data1"] = np.ones(mesh.n_points)
    mesh["scalar_data2"] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None

    # Raises if multiple scalar-like data with same name
    mesh["scalar_data"] = np.ones(mesh.n_points)
    mesh["scalar_data"] = np.ones(mesh.n_cells)
    mesh.set_active_scalars(None)
    with pytest.raises(AmbiguousDataError):
        pyvista.set_default_active_scalars(mesh)
    assert mesh.active_scalars_name is None


def test_vtk_points_deep_shallow():
    points = np.array([[0.0, 0.0, 0.0]])
    vtk_points = pyvista.vtk_points(points, deep=False)

    assert vtk_points.GetNumberOfPoints() == 1
    assert np.array_equal(vtk_points.GetPoint(0), points[0])

    # test shallow copy
    vtk_points.SetPoint(0, [1.0, 1.0, 1.0])

    assert np.array_equal(vtk_points.GetPoint(0), points[0])
    assert np.array_equal(vtk_points.GetPoint(0), [1.0, 1.0, 1.0])

    # test deep copy

    points = np.array([[0.0, 0.0, 0.0]])
    vtk_points = pyvista.vtk_points(points, deep=True)

    vtk_points.SetPoint(0, [1.0, 1.0, 1.0])

    assert not np.array_equal(vtk_points.GetPoint(0), points[0])
    assert np.array_equal(points[0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("force_float,expected_data_type", [(False, np.int64), (True, np.float32)])
def test_vtk_points_force_float(force_float, expected_data_type):
    np_points = np.array([[1, 2, 3]], dtype=np.int64)
    if force_float:
        with pytest.warns(UserWarning, match='Points is not a float type'):
            vtk_points = pyvista.vtk_points(np_points, force_float=force_float)
    else:
        vtk_points = pyvista.vtk_points(np_points, force_float=force_float)
    as_numpy = numpy_support.vtk_to_numpy(vtk_points.GetData())

    assert as_numpy.dtype == expected_data_type
