import numpy as np

import pyvista


def test_cylinder():
    surf = pyvista.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)

def test_cylinder_structured():
    cyl = pyvista.CylinderStructured()
    assert np.any(cyl.points)
    assert np.any(cyl.n_cells)


def test_arrow():
    surf = pyvista.Arrow([0, 0, 0], [1, 1, 1])
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_sphere():
    surf = pyvista.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_plane():
    surf = pyvista.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_line():
    line = pyvista.Line((0,0,0), (10, 1., 3))
    assert line.n_points == 2
    assert line.n_cells == 1
    line = pyvista.Line((0,0,0), (10, 1., 3), 10)
    assert line.n_points == 11
    assert line.n_cells == 1


def test_cube():
    cube = pyvista.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.,3., 5.,6., 7.,8.)
    cube = pyvista.Cube(bounds=bounds)
    assert np.any(cube.points)
    assert np.any(cube.faces)
    assert np.allclose(cube.bounds, bounds)


def test_cone():
    cone = pyvista.Cone()
    assert np.any(cone.points)
    assert np.any(cone.faces)


def test_box():
    geom = pyvista.Box()
    assert np.any(geom.points)


def test_polygon():
    geom = pyvista.Polygon()
    assert np.any(geom.points)


def test_disc():
    geom = pyvista.Disc()
    assert np.any(geom.points)


# def test_supertoroid():
#     geom = pyvista.SuperToroid()
#     assert np.any(geom.points)


# def test_ellipsoid():
#     geom = pyvista.Ellipsoid()
#     assert np.any(geom.points)


def test_text_3d():
    mesh = pyvista.Text3D("foo")
    assert mesh.n_points
    assert mesh.n_cells

def test_wavelet():
    mesh = pyvista.Wavelet()
    assert mesh.n_points
    assert mesh.n_cells
