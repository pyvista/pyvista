from itertools import permutations

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.geometric_objects import translate


def test_cylinder():
    surf = pv.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_cylinder_structured():
    cyl = pv.CylinderStructured()
    assert np.any(cyl.points)
    assert np.any(cyl.n_cells)


@pytest.mark.parametrize('scale', [None, 2.0, 4, 'auto'])
def test_arrow(scale):
    surf = pv.Arrow([0, 0, 0], [1, 1, 1], scale=scale)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow_raises_error():
    with pytest.raises(TypeError):
        pv.Arrow([0, 0, 0], [1, 1, 1], scale='badarg')


def test_sphere():
    surf = pv.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


@pytest.mark.parametrize(
    'expected', [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
)
def test_sphere_direction_points(expected):
    # from south pole to north pole
    north_pole = pv.Sphere(direction=expected, start_phi=0, end_phi=0).points[0]
    south_pole = pv.Sphere(direction=expected, start_phi=180, end_phi=180).points[0]
    actual = north_pole - south_pole
    assert np.array_equal(expected, actual)


def test_sphere_phi():
    atol = 1e-16
    north_hemisphere = pv.Sphere(start_phi=0, end_phi=90)
    assert np.all(north_hemisphere.points[:, 2] >= -atol)  # north is above XY plane
    south_hemisphere = pv.Sphere(start_phi=90, end_phi=180)
    assert np.all(south_hemisphere.points[:, 2] <= atol)  # south is below XY plane


def test_sphere_theta():
    atol = 1e-16

    quadrant1 = pv.Sphere(start_theta=0, end_theta=90)
    assert np.all(quadrant1.points[:, 0] >= -atol)  # +X
    assert np.all(quadrant1.points[:, 1] >= -atol)  # +Y

    quadrant2 = pv.Sphere(start_theta=90, end_theta=180)
    assert np.all(quadrant2.points[:, 0] <= atol)  # -X
    assert np.all(quadrant2.points[:, 1] >= -atol)  # +Y

    quadrant3 = pv.Sphere(start_theta=180, end_theta=270)
    assert np.all(quadrant3.points[:, 0] <= atol)  # -X
    assert np.all(quadrant3.points[:, 1] <= atol)  # -Y

    quadrant4 = pv.Sphere(start_theta=270, end_theta=360)
    assert np.all(quadrant4.points[:, 0] >= -atol)  # +X
    assert np.all(quadrant4.points[:, 1] <= atol)  # -Y


def test_sphere_unstructured():
    sphere = pv.SphereUnstructured()
    assert isinstance(sphere, pv.UnstructuredGrid)
    assert np.any(sphere.points)

    # make sure cell creation gives positive volume.
    for i, cell in enumerate(sphere.cell):
        assert cell.cast_to_unstructured_grid().volume > 0
    sphere = pv.SphereUnstructured(radius_resolution=5, theta_resolution=100, phi_resolution=100)
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3, rel=1e-3)


def test_sphere_unstructured_hollow():
    sphere = pv.SphereUnstructured(
        outer_radius=1.0,
        inner_radius=0.5,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * (1.0**3 - 0.5**3), rel=1e-3)


def test_sphere_unstructured_sequences():
    sphere = pv.SphereUnstructured(radius_resolution=5, theta_resolution=11, phi_resolution=13)
    sphere_seq = pv.SphereUnstructured(
        radius=np.linspace(0, 0.5, 5), theta=np.linspace(0, 360, 11), phi=np.linspace(0, 180, 13)
    )
    assert sphere == sphere_seq


def test_sphere_unstructured_theta_start_end():
    sphere = pv.SphereUnstructured(
        start_theta=0, end_theta=180, radius_resolution=5, theta_resolution=100, phi_resolution=100
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)

    sphere = pv.SphereUnstructured(
        start_theta=180,
        end_theta=360,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)

    sphere = pv.SphereUnstructured(
        start_theta=90, end_theta=120, radius_resolution=5, theta_resolution=100, phi_resolution=100
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 12, rel=1e-3)


def test_sphere_unstructured_phi_start_end():
    sphere = pv.SphereUnstructured(
        start_phi=0, end_phi=90, radius_resolution=5, theta_resolution=100, phi_resolution=100
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)

    sphere = pv.SphereUnstructured(
        start_phi=90,
        end_phi=180,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)


def test_plane():
    surf = pv.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)
    assert np.array_equal(surf.center, (0, 0, 0))


@pytest.mark.parametrize(
    'expected', [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
)
def test_plane_direction(expected):
    surf = pv.Plane(direction=expected)
    actual = surf.point_normals[0]
    assert np.array_equal(actual, expected)


def test_plane_size():
    i_sz = 2
    j_sz = 3
    surf = pv.Plane(i_size=i_sz, j_size=j_sz)
    assert np.array_equal(surf.bounds, (-i_sz / 2, i_sz / 2, -j_sz / 2, j_sz / 2, 0.0, 0.0))


def test_line():
    pointa = (0, 0, 0)
    pointb = (10, 1.0, 3)

    line = pv.Line(pointa, pointb)
    assert line.n_points == 2
    assert line.n_cells == 1
    line = pv.Line(pointa, pointb, 10)
    assert line.n_points == 11
    assert line.n_cells == 1

    with pytest.raises(ValueError):
        pv.Line(pointa, pointb, -1)

    with pytest.raises(TypeError):
        pv.Line(pointa, pointb, 0.1)  # from vtk

    with pytest.raises(TypeError):
        pv.Line((0, 0), pointb)

    with pytest.raises(TypeError):
        pv.Line(pointa, (10, 1.0))


def test_multiple_lines():
    points = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 2], [3, 3, 0]])
    multiple_lines = pv.MultipleLines(points=points)
    assert multiple_lines.n_points == 4
    assert multiple_lines.n_cells == 1

    points = np.array([[0, 0, 0], [1, 1 * np.sqrt(3), 0], [2, 0, 0], [3, 3 * np.sqrt(3), 0]])
    multiple_lines = pv.MultipleLines(points=points)

    with pytest.raises(ValueError):
        pv.MultipleLines(points[:, :1])

    with pytest.raises(ValueError):
        pv.MultipleLines(points[0, :])


def test_tube():
    pointa = (0, 0, 0)
    pointb = (10, 1.0, 3)

    tube = pv.Tube(n_sides=3)
    assert tube.n_points == 6
    assert tube.n_cells == 3
    tube = pv.Tube(pointa, pointb, 10)
    assert tube.n_points == 165
    assert tube.n_cells == 15

    with pytest.raises(ValueError):
        pv.Tube(pointa, pointb, -1)

    with pytest.raises(TypeError):
        pv.Tube(pointa, pointb, 0.1)  # from vtk

    with pytest.raises(TypeError):
        pv.Tube((0, 0), pointb)

    with pytest.raises(TypeError):
        pv.Tube(pointa, (10, 1.0))


def test_cube():
    cube = pv.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.0, 3.0, 5.0, 6.0, 7.0, 8.0)
    cube = pv.Cube(bounds=bounds)
    assert np.any(cube.points)
    assert np.any(cube.faces)
    assert np.allclose(cube.bounds, bounds)


def test_cone():
    cone = pv.Cone()
    assert np.any(cone.points)
    assert np.any(cone.faces)


def test_box():
    geom = pv.Box()
    assert np.any(geom.points)

    bounds = [-10.0, 10.0, 10.0, 20.0, -5.0, 5.0]
    level = 3
    quads = True
    mesh1 = pv.Box(bounds, level, quads)
    assert mesh1.n_cells == (level + 1) * (level + 1) * 6
    assert np.allclose(mesh1.bounds, bounds)

    quads = False
    mesh2 = pv.Box(bounds, level, quads)
    assert mesh2.n_cells == mesh1.n_cells * 2


def test_polygon():
    geom = pv.Polygon()
    assert np.any(geom.points)

    geom1 = pv.Polygon(fill=True)
    assert geom1.n_cells == 2
    geom2 = pv.Polygon(fill=False)
    assert geom2.n_cells == 1


def test_disc():
    geom = pv.Disc()
    assert np.any(geom.points)

    normal = np.array([1.2, 3.4, 5.6])
    unit_normal = normal / np.linalg.norm(normal)
    geom = pv.Disc(normal=unit_normal)

    normals = geom.compute_normals()['Normals']
    assert np.allclose(np.dot(normals, unit_normal), 1)

    center = (1.2, 3.4, 5.6)
    geom = pv.Disc(center=center)

    assert np.allclose(geom.bounds, pv.Disc().bounds + np.array([1.2, 1.2, 3.4, 3.4, 5.6, 5.6]))


def test_superquadric():
    geom = pv.Superquadric()
    assert np.any(geom.points)


# def test_supertoroid():
#     geom = pv.SuperToroid()
#     assert np.any(geom.points)


# def test_ellipsoid():
#     geom = pv.Ellipsoid()
#     assert np.any(geom.points)


def test_text_3d():
    mesh = pv.Text3D("foo")
    assert mesh.n_points
    assert mesh.n_cells


def test_wavelet():
    mesh = pv.Wavelet()
    assert mesh.n_points
    assert mesh.n_cells


def test_circular_arc():
    pointa = [-1, 0, 0]
    pointb = [0, 1, 0]
    center = [0, 0, 0]
    resolution = 100

    mesh = pv.CircularArc(pointa, pointb, center, resolution)
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01) * np.pi / 2.0
    assert np.allclose(mesh['Distance'], distance)

    # pointa and pointb are not equidistant from center
    with pytest.raises(ValueError):
        mesh = pv.CircularArc([-1, 0, 0], [-0.99, 0.001, 0], [0, 0, 0], 100)


def test_circular_arc_from_normal():
    center = [0, 0, 0]
    normal = [0, 0, 1]
    polar = [-2.0, 0, 0]
    angle = 90
    resolution = 100

    mesh = pv.CircularArcFromNormal(center, resolution, normal, polar, angle)
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

    mesh = pv.Pyramid(points)
    assert mesh.n_points
    assert mesh.n_cells
    assert np.allclose(mesh.points, points)

    # test pyramid with default points
    mesh = pv.Pyramid()
    assert isinstance(mesh, pv.UnstructuredGrid)


def test_triangle():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    points = np.array([pointa, pointb, pointc])

    mesh = pv.Triangle(points)
    assert mesh.n_points
    assert mesh.n_cells
    assert np.allclose(mesh.points, points)


def test_quadrilateral():
    pointa = [1.0, 1.0, 1.0]
    pointb = [-1.0, 1.0, 1.0]
    pointc = [-1.0, -1.0, 1.0]
    pointd = [1.0, -1.0, 1.0]
    points = np.array([pointa, pointb, pointc, pointd])

    mesh = pv.Quadrilateral(points)
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
        pv.core.errors.PyVistaDeprecationWarning,
        match='Please use ``pyvista.Quadrilateral``.',
    ):
        mesh = pv.Rectangle(points)
        assert mesh.n_points
        assert mesh.n_cells
        assert np.allclose(mesh.points, points)


def test_rectangle():
    pointa = [3.0, 1.0, 1.0]
    pointb = [3.0, 2.0, 1.0]
    pointc = [1.0, 2.0, 1.0]
    pointd = [1.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pv.core.utilities.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pv.core.utilities.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc, pointd])
    )

    # Test all possible orders of the points
    for pt_tuples in permutations(rotated, 4):
        mesh = pv.Rectangle(list(pt_tuples[0:3]))
        assert mesh.n_points
        assert mesh.n_cells
        assert np.allclose(mesh.points, pt_tuples)


def test_rectangle_not_orthognal_entries():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]
    pointc = [1.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pv.core.utilities.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pv.core.utilities.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc])
    )

    with pytest.raises(ValueError, match="The three points should defined orthogonal vectors"):
        pv.Rectangle(rotated)


def test_rectangle_two_identical_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]
    pointc = [3.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pv.core.utilities.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pv.core.utilities.transformations.apply_transformation_to_points(
        trans, np.array([pointa, pointb, pointc])
    )

    with pytest.raises(
        ValueError, match="Unable to build a rectangle with less than three different points"
    ):
        pv.Rectangle(rotated)


def test_rectangle_not_enough_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]

    with pytest.raises(TypeError, match='Points must be given as length 3 np.ndarray or list'):
        pv.Rectangle([pointa, pointb])


def test_circle():
    radius = 1.0

    mesh = pv.Circle(radius)
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

    mesh = pv.Ellipse(semi_major_axis, semi_minor_axis)
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
    solid_from_str = pv.PlatonicSolid(kind_str)
    solid_from_int = pv.PlatonicSolid(kind_int)
    assert solid_from_str == solid_from_int

    # verify type of solid
    assert solid_from_str.n_points == n_vertices
    assert solid_from_str.n_faces == n_faces


def test_platonic_invalids():
    with pytest.raises(ValueError):
        pv.PlatonicSolid(kind='invalid')
    with pytest.raises(ValueError):
        pv.PlatonicSolid(kind=42)
    with pytest.raises(ValueError):
        pv.PlatonicSolid(kind=[])


def test_tetrahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Tetrahedron(radius=radius, center=center)
    assert solid.n_points == 4
    assert solid.n_faces == 4
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Tetrahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_octahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Octahedron(radius=radius, center=center)
    assert solid.n_points == 6
    assert solid.n_faces == 8
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Octahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_dodecahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Dodecahedron(radius=radius, center=center)
    assert solid.n_points == 20
    assert solid.n_faces == 12
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Dodecahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_icosahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Icosahedron(radius=radius, center=center)
    assert solid.n_points == 12
    assert solid.n_faces == 20
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Icosahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_icosphere():
    center = (1.0, 2.0, 3.0)
    radius = 2.4
    nsub = 2
    icosphere = pv.Icosphere(radius=radius, center=center, nsub=nsub)
    assert np.allclose(icosphere.center, center)
    assert np.allclose(np.linalg.norm(icosphere.points - icosphere.center, axis=1), radius)

    icosahedron = pv.Icosahedron()
    assert icosahedron.n_faces * 4**nsub == icosphere.n_faces


@pytest.fixture()
def bunny():
    return examples.download_bunny()


@pytest.mark.parametrize("is_negative", (True, False))
@pytest.mark.parametrize("delta", ([0, 0, 0], [1e-8, 0, 0], [0, 0, 1e-8]))
def test_translate_direction_collinear(is_negative, delta, bunny):
    mesh_in = bunny
    direction = np.array([0.0, 1.0, 0.0]) + delta
    if is_negative:
        direction *= -1
    mesh_out = mesh_in.copy()
    translate(mesh_out, direction=direction)
    points_in = mesh_in.points
    points_out = mesh_out.points

    if is_negative:
        assert np.allclose(points_in[:, 0], -points_out[:, 1])
        assert np.allclose(points_in[:, 1], points_out[:, 0])
        assert np.allclose(points_in[:, 2], points_out[:, 2])
    else:
        assert np.allclose(points_in[:, 0], points_out[:, 1])
        assert np.allclose(points_in[:, 1], -points_out[:, 0])
        assert np.allclose(points_in[:, 2], points_out[:, 2])
