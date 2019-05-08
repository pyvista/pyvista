import numpy as np

import vista


def test_cylinder():
    surf = vista.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow():
    surf = vista.Arrow([0, 0, 0], [1, 1, 1])
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_sphere():
    surf = vista.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_plane():
    surf = vista.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_line():
    line = vista.Line((0,0,0), (10, 1., 3))
    assert line.n_points == 2
    assert line.n_cells == 1
    line = vista.Line((0,0,0), (10, 1., 3), 10)
    assert line.n_points == 11
    assert line.n_cells == 1


def test_cube():
    cube = vista.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.,3., 5.,6., 7.,8.)
    cube = vista.Cube(bounds=bounds)
    assert np.any(cube.points)
    assert np.any(cube.faces)
    assert np.allclose(cube.bounds, bounds)


def test_cone():
    cone = vista.Cone()
    assert np.any(cone.points)
    assert np.any(cone.faces)


def test_box():
    geom = vista.Box()
    assert np.any(geom.points)


def test_polygon():
    geom = vista.Polygon()
    assert np.any(geom.points)


def test_disc():
    geom = vista.Disc()
    assert np.any(geom.points)
