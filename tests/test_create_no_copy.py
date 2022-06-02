import numpy as np
import pytest

import pyvista

pointsetmark = pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)


@pytest.fixture
def structured_points():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    source = np.empty((x.size, 3), x.dtype)
    source[:, 0] = x.ravel('F')
    source[:, 1] = y.ravel('F')
    source[:, 2] = z.ravel('F')
    return source, (*x.shape, 1)


def test_no_copy_polydata_init():
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_polydata_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_init(structured_points):
    source, dims = structured_points
    mesh = pyvista.StructuredGrid(source)
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_points_setter(structured_points):
    source, dims = structured_points
    mesh = pyvista.StructuredGrid()
    mesh.points = source
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


@pointsetmark
def test_no_copy_pointset_init():
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


@pointsetmark
def test_no_copy_pointset_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_unstructured_grid_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.UnstructuredGrid()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_rectilinear_grid():
    xrng = np.arange(-10, 10, 2, dtype=float)
    yrng = np.arange(-10, 10, 5, dtype=float)
    zrng = np.arange(-10, 10, 1, dtype=float)
    mesh = pyvista.RectilinearGrid(xrng, yrng, zrng)
    x = mesh.x
    x /= 2
    assert np.array_equal(mesh.x, x)
    assert np.may_share_memory(mesh.x, x)
    assert np.array_equal(mesh.x, xrng)
    assert np.may_share_memory(mesh.x, xrng)
    y = mesh.y
    y /= 2
    assert np.array_equal(mesh.y, y)
    assert np.may_share_memory(mesh.y, y)
    assert np.array_equal(mesh.y, yrng)
    assert np.may_share_memory(mesh.y, yrng)
    z = mesh.z
    z /= 2
    assert np.array_equal(mesh.z, z)
    assert np.may_share_memory(mesh.z, z)
    assert np.array_equal(mesh.z, zrng)
    assert np.may_share_memory(mesh.z, zrng)


def test_raise_rectilinear_grid_non_unique():
    rng_uniq = np.arange(4.0)
    rng_dupe = np.array([0, 1, 2, 2], dtype=float)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_uniq, rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_uniq, rng_uniq, rng_dupe, check_duplicates=True)
