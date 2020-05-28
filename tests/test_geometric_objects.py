import numpy as np
import pytest

import pyvista


def test_cylinder():
    surf = pyvista.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_cylinder_structured():
    cyl = pyvista.CylinderStructured()
    assert np.any(cyl.points)
    assert np.any(cyl.n_cells)


@pytest.mark.parametrize('scale', [None, 2.0, 4, 'auto'])
def test_arrow(scale):
    surf = pyvista.Arrow([0, 0, 0], [1, 1, 1], scale=scale)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow_raises_error():
    with pytest.raises(TypeError):
        surf = pyvista.Arrow([0, 0, 0], [1, 1, 1], scale='badarg')


def test_sphere():
    surf = pyvista.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_plane():
    surf = pyvista.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_line():
    pointa = (0, 0, 0)
    pointb = (10, 1., 3)

    line = pyvista.Line(pointa, pointb)
    assert line.n_points == 2
    assert line.n_cells == 1
    line = pyvista.Line(pointa, pointb, 10)
    assert line.n_points == 11
    assert line.n_cells == 1

    with pytest.raises(ValueError):
        pyvista.Line(pointa, pointb, -1)

    with pytest.raises(TypeError):
        pyvista.Line(pointa, pointb, 0.1) # from vtk


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


def test_circular_arc():
    pointa = [-1, 0, 0]
    pointb = [0, 1, 0]
    center = [0, 0, 0]
    resolution = 100

    mesh = pyvista.CircularArc(pointa, pointb, center, resolution)
    assert mesh.n_points
    assert mesh.n_cells

    mesh = pyvista.CircularArc([-1, 0, 0], [0, 0, 1], [0, 0, 0], normal=[0, 0, 1],
                               polar=[1, 0, 1], negative=True, angle=180)
    assert mesh.n_points
    assert mesh.n_cells
