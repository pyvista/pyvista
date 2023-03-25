from itertools import permutations

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
        pyvista.Arrow([0, 0, 0], [1, 1, 1], scale='badarg')


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
    pointb = (10, 1.0, 3)

    line = pyvista.Line(pointa, pointb)
    assert line.n_points == 2
    assert line.n_cells == 1
    line = pyvista.Line(pointa, pointb, 10)
    assert line.n_points == 11
    assert line.n_cells == 1

    with pytest.raises(ValueError):
        pyvista.Line(pointa, pointb, -1)

    with pytest.raises(TypeError):
        pyvista.Line(pointa, pointb, 0.1)  # from vtk

    with pytest.raises(TypeError):
        pyvista.Line((0, 0), pointb)

    with pytest.raises(TypeError):
        pyvista.Line(pointa, (10, 1.0))


def test_multiple_lines():
    points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 2], [3, 3, 0]])
    multiple_lines = pyvista.MultipleLines(points=points)
    assert multiple_lines.n_points == 4
    assert multiple_lines.n_cells == 1

    points = np.array([[0, 0, 0], [1, 1 * np.sqrt(3), 0], [2, 0, 0], [3, 3 * np.sqrt(3), 0]])
    multiple_lines = pyvista.MultipleLines(points=points)

    with pytest.raises(ValueError):
        pyvista.MultipleLines(points[:, :1])

    with pytest.raises(ValueError):
        pyvista.MultipleLines(points[0, :])


def test_tube():
    pointa = (0, 0, 0)
    pointb = (10, 1.0, 3)

    tube = pyvista.Tube(n_sides=3)
    assert tube.n_points == 6
    assert tube.n_cells == 3
    tube = pyvista.Tube(pointa, pointb, 10)
    assert tube.n_points == 165
    assert tube.n_cells == 15

    with pytest.raises(ValueError):
        pyvista.Tube(pointa, pointb, -1)

    with pytest.raises(TypeError):
        pyvista.Tube(pointa, pointb, 0.1)  # from vtk

    with pytest.raises(TypeError):
        pyvista.Tube((0, 0), pointb)

    with pytest.raises(TypeError):
        pyvista.Tube(pointa, (10, 1.0))


def test_cube():
    cube = pyvista.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.0, 3.0, 5.0, 6.0, 7.0, 8.0)
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
    assert mesh2.n_cells == mesh1.n_cells * 2


def test_polygon():
    geom = pyvista.Polygon()
    assert np.any(geom.points)

    geom1 = pyvista.Polygon(fill=True)
    assert geom1.n_cells == 2
    geom2 = pyvista.Polygon(fill=False)
    assert geom2.n_cells == 1


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


def test_superquadric():
    geom = pyvista.Superquadric()
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
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01) * np.pi / 2.0
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
    distance = np.arange(0.0, 1.0 + 0.01, 0.01) * np.pi
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

    # test pyramid with default points
    mesh = pyvista.Pyramid()
    assert isinstance(mesh, pyvista.UnstructuredGrid)


def test_triangle():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    points = np.array([pointa, pointb, pointc])

    mesh = pyvista.Triangle(points)
    assert mesh.n_points
    assert mesh.n_cells
    assert np.allclose(mesh.points, points)


def test_quadrilateral():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    pointd = [1.0, -1.0, 1.0]
    points = np.array([pointa, pointb, pointc, pointd])

    mesh = pyvista.Quadrilateral(points)
    assert mesh.n_points
    assert mesh.n_cells
    assert np.allclose(mesh.points, points)


def test_rectangle_4points_deprecation():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    pointd = [1.0, -1.0, 1.0]
    points = np.array([pointa, pointb, pointc, pointd])

    with pytest.warns(
        pyvista.utilities.misc.PyVistaDeprecationWarning,
        match='Please use ``pyvista.Quadrilateral``.',
    ):
        mesh = pyvista.Rectangle(points)
        assert mesh.n_points
        assert mesh.n_cells
        assert np.allclose(mesh.points, points)


def test_rectangle():
    pointa = [3.0, 1.0, 1.0]
    pointb = [3.0, 2.0, 1.0]
    pointc = [1.0, 2.0, 1.0]
    pointd = [1.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pyvista.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pyvista.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc, pointd])
    )

    # Test all possible orders of the points
    for pt_tuples in permutations(rotated, 4):
        mesh = pyvista.Rectangle(list(pt_tuples[0:3]))
        assert mesh.n_points
        assert mesh.n_cells
        assert np.allclose(mesh.points, pt_tuples)


def test_rectangle_not_orthognal_entries():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]
    pointc = [1.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pyvista.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pyvista.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc])
    )

    with pytest.raises(ValueError, match="The three points should defined orthogonal vectors"):
        pyvista.Rectangle(rotated)


def test_rectangle_two_identical_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]
    pointc = [3.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pyvista.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pyvista.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc])
    )

    with pytest.raises(
        ValueError, match="Unable to build a rectangle with less than three different points"
    ):
        pyvista.Rectangle(rotated)


def test_rectangle_not_enough_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]

    with pytest.raises(TypeError, match='Points must be given as length 3 np.ndarray or list'):
        pyvista.Rectangle([pointa, pointb])


def test_circle():
    radius = 1.0

    mesh = pyvista.Circle(radius)
    assert mesh.n_points
    assert mesh.n_cells
    diameter = np.max(mesh.points[:, 0]) - np.min(mesh.points[:, 0])
    assert np.isclose(diameter, radius * 2.0, rtol=1e-3)
    line_lengths = np.linalg.norm(
        np.roll(mesh.points, shift=1, axis=0) - mesh.points,
        axis=1,
    )
    assert np.allclose(line_lengths[0], line_lengths)


def test_ellipse():
    semi_major_axis = 8.0
    semi_minor_axis = 4.0

    mesh = pyvista.Ellipse(semi_major_axis, semi_minor_axis)
    assert mesh.n_points
    assert mesh.n_cells
    major_axis_diameter = np.max(mesh.points[:, 0]) - np.min(mesh.points[:, 0])
    minor_axis_diameter = np.max(mesh.points[:, 1]) - np.min(mesh.points[:, 1])
    assert np.isclose(major_axis_diameter, semi_major_axis * 2.0, rtol=1e-3)
    assert np.isclose(minor_axis_diameter, semi_minor_axis * 2.0, rtol=1e-3)


@pytest.mark.parametrize(
    'kind_str, kind_int, n_vertices, n_faces',
    zip(
        ['tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron'],
        range(5),
        [4, 8, 6, 12, 20],
        [4, 6, 8, 20, 12],
    ),
)
def test_platonic_solids(kind_str, kind_int, n_vertices, n_faces):
    # verify integer mapping
    solid_from_str = pyvista.PlatonicSolid(kind_str)
    solid_from_int = pyvista.PlatonicSolid(kind_int)
    assert solid_from_str == solid_from_int

    # verify type of solid
    assert solid_from_str.n_points == n_vertices
    assert solid_from_str.n_faces == n_faces


def test_platonic_invalids():
    with pytest.raises(ValueError):
        pyvista.PlatonicSolid(kind='invalid')
    with pytest.raises(ValueError):
        pyvista.PlatonicSolid(kind=42)
    with pytest.raises(ValueError):
        pyvista.PlatonicSolid(kind=[])


def test_tetrahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pyvista.Tetrahedron(radius=radius, center=center)
    assert solid.n_points == 4
    assert solid.n_faces == 4
    assert np.allclose(solid.center, center)

    doubled_solid = pyvista.Tetrahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_octahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pyvista.Octahedron(radius=radius, center=center)
    assert solid.n_points == 6
    assert solid.n_faces == 8
    assert np.allclose(solid.center, center)

    doubled_solid = pyvista.Octahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_dodecahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pyvista.Dodecahedron(radius=radius, center=center)
    assert solid.n_points == 20
    assert solid.n_faces == 12
    assert np.allclose(solid.center, center)

    doubled_solid = pyvista.Dodecahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_icosahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pyvista.Icosahedron(radius=radius, center=center)
    assert solid.n_points == 12
    assert solid.n_faces == 20
    assert np.allclose(solid.center, center)

    doubled_solid = pyvista.Icosahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)
