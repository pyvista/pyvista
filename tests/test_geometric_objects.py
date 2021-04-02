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

    bounds = [-10.0, 10.0, 10.0, 20.0, -5.0, 5.0]
    level = 3
    quads = True
    mesh1 = pyvista.Box(bounds, level, quads)
    assert mesh1.n_cells == (level + 1) * (level + 1) * 6
    assert np.allclose(mesh1.bounds, bounds)

    quads = False
    mesh2 = pyvista.Box(bounds, level, quads)
    assert mesh2.n_cells == mesh1.n_cells*2


def test_polygon():
    geom = pyvista.Polygon()
    assert np.any(geom.points)


def test_disc():
    geom = pyvista.Disc()
    assert np.any(geom.points)

    normal = np.array([1.2, 3.4, 5.6])
    unit_normal = normal / np.linalg.norm(normal)
    geom = pyvista.Disc(normal=unit_normal)

    normals = geom.compute_normals()['Normals']
    assert np.allclose(np.dot(normals, unit_normal), 1)

    center = (1.2, 3.4, 5.6)
    geom = pyvista.Disc(center=center)

    assert np.allclose(
        geom.bounds, pyvista.Disc().bounds + np.array([1.2, 1.2, 3.4, 3.4, 5.6, 5.6])
    )


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
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01)*np.pi/2.0
    assert np.allclose(mesh['Distance'], distance)

    # pointa and pointb are not equidistant from center
    with pytest.raises(ValueError):
        mesh = pyvista.CircularArc([-1, 0, 0], [-0.99, 0.001, 0], [0, 0, 0], 100)


def test_circular_arc_from_normal():
    center = [0, 0, 0]
    normal = [0, 0, 1]
    polar = [-2.0, 0, 0]
    angle = 90
    resolution = 100

    mesh = pyvista.CircularArcFromNormal(center, resolution, normal, polar, angle)
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01)*np.pi
    assert np.allclose(mesh['Distance'], distance)


def test_pyramid():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    pointd = [1.0, -1.0, 1.0]
    pointe = [0.0, 0.0, 0.0]
    points = np.array([pointa, pointb, pointc, pointd, pointe])

    mesh = pyvista.Pyramid(points)
    assert mesh.n_points
    assert mesh.n_cells
    assert np.allclose(mesh.points, points)
