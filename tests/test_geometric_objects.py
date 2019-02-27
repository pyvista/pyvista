import numpy as np

import vtki


def test_cylinder():
    surf = vtki.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow():
    surf = vtki.Arrow([0, 0, 0], [1, 1, 1])
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_sphere():
    surf = vtki.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_plane():
    surf = vtki.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_line():
    line = vtki.Line((0,0,0), (10, 1., 3))
    assert line.n_points == 2
    assert line.n_cells == 1
    line = vtki.Line((0,0,0), (10, 1., 3), 10)
    assert line.n_points == 11
    assert line.n_cells == 1


def test_cube():
    cube = vtki.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.,3., 5.,6., 7.,8.)
    cube = vtki.Cube(bounds=bounds)
    assert np.any(cube.points)
    assert np.any(cube.faces)
    assert np.allclose(cube.bounds, bounds)


def test_cone():
    cone = vtki.Cone()
    assert np.any(cone.points)
    assert np.any(cone.faces)
