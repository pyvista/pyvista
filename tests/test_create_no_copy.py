import numpy as np
import pytest

import pyvista


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
    return source


def test_no_copy_polydata_init():
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData(source)
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


def test_no_copy_polydata_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


def test_no_copy_structured_mesh_init(structured_points):
    source = structured_points
    mesh = pyvista.StructuredGrid(source)
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


def test_no_copy_structured_mesh_points_setter(structured_points):
    source = structured_points
    mesh = pyvista.StructuredGrid()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


def test_no_copy_pointset_init():
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet(source)
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


def test_no_copy_pointset_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)


# def test_no_copy_unstructured_grid_init():
#     ...  # TODO


def test_no_copy_unstructured_grid_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.UnstructuredGrid()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.allclose(mesh.points, pts)
    assert np.allclose(mesh.points, source)
