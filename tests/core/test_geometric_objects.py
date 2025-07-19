from __future__ import annotations

from itertools import permutations
import re

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

import pyvista as pv


@given(points=st.lists(elements=st.integers()).filter(lambda x: len(x) != 5))
def test_pyramid_raises(points):
    with pytest.raises(
        TypeError, match=re.escape('Points must be given as length 5 np.ndarray or list.')
    ):
        pv.Pyramid(points=points)


@given(points=st.lists(elements=st.integers()).filter(lambda x: len(x) != 3))
def test_triangle_raises(points):
    with pytest.raises(
        TypeError, match=re.escape('Points must be given as length 3 np.ndarray or list')
    ):
        pv.Triangle(points=points)


@given(points=st.lists(elements=st.integers()).filter(lambda x: len(x) != 4))
def test_quadrilateral_raises(points):
    with pytest.raises(
        TypeError, match=re.escape('Points must be given as length 4 np.ndarray or list')
    ):
        pv.Quadrilateral(points=points)


def test_cylinder():
    surf = pv.Cylinder(center=[0, 10, 0], direction=[1, 1, 1], radius=1, height=5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_cylinder_structured():
    cyl = pv.CylinderStructured()
    assert np.any(cyl.points)
    assert np.any(cyl.n_cells)


@pytest.mark.parametrize('scale', [None, 2.0, 4, 'auto'])
def test_arrow(scale):
    surf = pv.Arrow(start=[0, 0, 0], direction=[1, 1, 1], scale=scale)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow_raises_error():
    with pytest.raises(TypeError):
        pv.Arrow(start=[0, 0, 0], direction=[1, 1, 1], scale='badarg')


def test_sphere():
    surf = pv.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


@pytest.mark.parametrize(
    'expected',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
)
def test_sphere_direction_points(expected):
    # from south pole to north pole
    north_pole = pv.Sphere(direction=expected, start_phi=0, end_phi=0).points[0]
    south_pole = pv.Sphere(direction=expected, start_phi=180, end_phi=180).points[0]
    actual = north_pole - south_pole
    assert np.array_equal(expected, actual)


# test_sphere_phi and test_sphere_theta are similar to ones for SolidSphere
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


def test_solid_sphere():
    sphere = pv.SolidSphere()
    assert isinstance(sphere, pv.UnstructuredGrid)
    assert np.any(sphere.points)

    # make sure cell creation gives positive volume.
    for cell in sphere.cell:
        assert cell.cast_to_unstructured_grid().volume > 0
    sphere = pv.SolidSphere(radius_resolution=5, theta_resolution=100, phi_resolution=100)
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3, rel=1e-3)


def test_solid_sphere_hollow():
    sphere = pv.SolidSphere(
        outer_radius=1.0,
        inner_radius=0.5,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * (1.0**3 - 0.5**3), rel=1e-3)


def test_solid_sphere_generic():
    sphere = pv.SolidSphere(radius_resolution=5, theta_resolution=11, phi_resolution=13)
    sphere_seq = pv.SolidSphereGeneric(
        radius=np.linspace(0, 0.5, 5),
        theta=np.linspace(0, 360, 11),
        phi=np.linspace(0, 180, 13),
    )
    assert sphere == sphere_seq


def test_solid_sphere_theta_start_end():
    sphere = pv.SolidSphere(
        start_theta=0,
        end_theta=180,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)

    sphere = pv.SolidSphere(
        start_theta=180,
        end_theta=360,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 2, rel=1e-3)

    sphere = pv.SolidSphere(
        start_theta=90,
        end_theta=120,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3 / 12, rel=1e-3)


def test_solid_sphere_phi_start_end():
    exp_sphere_volume = 4.0 / 3.0 * np.pi * 0.5**3

    sphere = pv.SolidSphere(
        start_phi=0,
        end_phi=90,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(exp_sphere_volume / 2, rel=1e-3)

    sphere = pv.SolidSphere(
        start_phi=90,
        end_phi=180,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )
    assert sphere.volume == pytest.approx(exp_sphere_volume / 2, rel=1e-3)

    sphere = pv.SolidSphere(
        start_phi=45,
        end_phi=135,
        radius_resolution=5,
        theta_resolution=100,
        phi_resolution=100,
    )

    hcone = 0.5 * np.sin(np.pi / 4)
    vcone = np.pi / 3 * hcone**3
    hcap = 0.5 - hcone
    vcap = np.pi / 3 * hcap**2 * (3 * 0.5 - hcap)

    assert sphere.volume == pytest.approx(exp_sphere_volume - 2 * (vcone + vcap), rel=1e-3)


def test_solid_sphere_resolution_edge_cases():
    sphere = pv.SolidSphere(radius_resolution=2)
    assert sphere.volume > 0

    sphere = pv.SolidSphere(radius_resolution=2, inner_radius=0.1)
    assert sphere.volume > 0

    sphere = pv.SolidSphere(theta_resolution=2, start_theta=45, end_theta=90)
    assert sphere.volume > 0

    sphere = pv.SolidSphere(phi_resolution=2, start_phi=45, end_phi=90)
    assert sphere.volume > 0


def test_solid_sphere_resolution_errors():
    with pytest.raises(ValueError, match='minimum radius cannot be negative'):
        pv.SolidSphere(inner_radius=-1)
    with pytest.raises(ValueError, match='max theta and min theta must be within 360 degrees'):
        pv.SolidSphere(start_theta=-1)
    with pytest.raises(ValueError, match='minimum phi cannot be negative'):
        pv.SolidSphere(start_phi=-1)
    with pytest.raises(ValueError, match='max theta and min theta must be within 360 degrees'):
        pv.SolidSphere(end_theta=370)
    with pytest.raises(ValueError, match='maximum phi cannot be > 180'):
        pv.SolidSphere(end_phi=190)
    with pytest.raises(
        ValueError,
        match=re.escape('max theta and min theta must be within 2 * np.pi'),
    ):
        pv.SolidSphere(end_theta=2.1 * np.pi, radians=True)
    with pytest.raises(ValueError, match='maximum phi cannot be > np.pi'):
        pv.SolidSphere(end_phi=1.1 * np.pi, radians=True)

    with pytest.raises(ValueError, match='radius is not monotonically increasing'):
        pv.SolidSphereGeneric(radius=(0, 10, 1))
    with pytest.raises(ValueError, match='theta is not monotonically increasing'):
        pv.SolidSphereGeneric(theta=(0, 180, 90))
    with pytest.raises(ValueError, match='phi is not monotonically increasing'):
        pv.SolidSphereGeneric(phi=(0, 180, 90))

    with pytest.raises(ValueError, match='radius resolution must be 2 or more'):
        pv.SolidSphere(radius_resolution=1)
    with pytest.raises(ValueError, match='theta resolution must be 2 or more'):
        pv.SolidSphere(theta_resolution=1)
    with pytest.raises(ValueError, match='phi resolution must be 2 or more'):
        pv.SolidSphere(phi_resolution=1)


# test_solid_sphere_phi and test_solid_sphere_theta are similar to ones for Sphere
def test_solid_sphere_phi():
    atol = 1e-16
    north_hemisphere = pv.SolidSphere(start_phi=0, end_phi=90)
    assert np.all(north_hemisphere.points[:, 2] >= -atol)  # north is above XY plane
    south_hemisphere = pv.SolidSphere(start_phi=90, end_phi=180)
    assert np.all(south_hemisphere.points[:, 2] <= atol)  # south is below XY plane


def test_solid_sphere_theta():
    atol = 1e-16

    quadrant1 = pv.SolidSphere(start_theta=0, end_theta=90)
    assert np.all(quadrant1.points[:, 0] >= -atol)  # +X
    assert np.all(quadrant1.points[:, 1] >= -atol)  # +Y

    quadrant2 = pv.SolidSphere(start_theta=90, end_theta=180)
    assert np.all(quadrant2.points[:, 0] <= atol)  # -X
    assert np.all(quadrant2.points[:, 1] >= -atol)  # +Y

    quadrant3 = pv.SolidSphere(start_theta=180, end_theta=270)
    assert np.all(quadrant3.points[:, 0] <= atol)  # -X
    assert np.all(quadrant3.points[:, 1] <= atol)  # -Y

    quadrant4 = pv.SolidSphere(start_theta=270, end_theta=360)
    assert np.all(quadrant4.points[:, 0] >= -atol)  # +X
    assert np.all(quadrant4.points[:, 1] <= atol)  # -Y


def test_solid_sphere_radians():
    deg = pv.SolidSphere()
    rad = pv.SolidSphere(radians=True)
    assert np.allclose(deg.points, rad.points)

    deg = pv.SolidSphere(start_theta=15, end_theta=180, start_phi=30, end_phi=90)
    rad = pv.SolidSphere(
        start_theta=np.deg2rad(15),
        end_theta=np.deg2rad(180),
        start_phi=np.deg2rad(30),
        end_phi=np.deg2rad(90),
        radians=True,
    )
    assert np.allclose(deg.points, rad.points)

    deg = pv.SolidSphereGeneric()
    rad = pv.SolidSphereGeneric(radians=True)
    assert np.allclose(deg.points, rad.points)

    theta = np.linspace(15, 180, 30)
    phi = np.linspace(30, 90, 30)
    deg = pv.SolidSphereGeneric(theta=theta, phi=phi)
    rad = pv.SolidSphereGeneric(theta=np.deg2rad(theta), phi=np.deg2rad(phi), radians=True)
    assert np.allclose(deg.points, rad.points)


def test_solid_sphere_theta_range():
    reference = pv.SolidSphere(start_theta=15, end_theta=105)
    pos = pv.SolidSphere(start_theta=15 + 720, end_theta=105 + 720)
    assert np.allclose(reference.points, pos.points)

    both_sides = pv.SolidSphere(start_theta=-45, end_theta=45)
    assert np.isclose(reference.volume, both_sides.volume)


def test_solid_sphere_tol_radius():
    solid_sphere = pv.SolidSphere(inner_radius=1e-5)
    assert np.array_equal(solid_sphere.points[0, :], [0.0, 0.0, 1.0e-5])

    solid_sphere = pv.SolidSphere(inner_radius=1e-10)
    assert np.array_equal(solid_sphere.points[0, :], [0.0, 0.0, 0.0])

    solid_sphere = pv.SolidSphere(inner_radius=1e-10, tol_radius=1e-11)
    assert np.array_equal(solid_sphere.points[0, :], [0.0, 0.0, 1.0e-10])


@pytest.mark.parametrize('radians', [True, False])
def test_solid_sphere_tol_angle(radians):
    max_phi = np.pi if radians else 180.0

    # when phi point not on axis, it is skipped in point ordering
    # When radius_resolution=2, there are a maximum of two axis points
    solid_sphere = pv.SolidSphere(start_phi=1e-3, radius_resolution=2, radians=radians)
    # start_phi is greater than tol, so the positive axis point is skipped
    assert np.allclose(solid_sphere.points[1, :], [0.0, 0.0, -0.5])
    # when end_phi is greater than tol, the negative axis point is skipped
    # that next points is above the z axis
    solid_sphere = pv.SolidSphere(end_phi=max_phi - 1e-3, radius_resolution=2, radians=radians)
    assert solid_sphere.points[2, 2] > 0.0

    solid_sphere = pv.SolidSphere(
        start_phi=1e-3,
        radius_resolution=2,
        radians=radians,
        tol_angle=1e-2,
    )
    # Positive axis point is there, but it is now slightly offset.
    assert np.allclose(solid_sphere.points[1, :], [0.0, 0.0, 0.5], atol=1e-3)
    # Negative axis point is there
    solid_sphere = pv.SolidSphere(
        end_phi=max_phi - 1e-3,
        radius_resolution=2,
        radians=radians,
        tol_angle=1e-2,
    )
    assert np.allclose(solid_sphere.points[2, :], [0.0, 0.0, -0.5], atol=1e-3)

    # When theta is not detected to overlap it will result in more points
    reference = pv.SolidSphere(radians=radians)
    solid_sphere = pv.SolidSphere(start_theta=1e-3, radians=radians)
    assert solid_sphere.n_points > reference.n_points
    solid_sphere = pv.SolidSphere(start_theta=1e-3, radians=radians, tol_angle=1e-1)
    assert solid_sphere.n_points == reference.n_points


def test_plane():
    surf = pv.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)
    assert np.array_equal(surf.center, (0, 0, 0))


@pytest.mark.parametrize(
    'expected',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
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
    line = pv.Line(pointa, pointb, resolution=10)
    assert line.n_points == 11
    assert line.n_cells == 1

    with pytest.raises(ValueError):  # noqa: PT011
        pv.Line(pointa, pointb, resolution=-1)

    with pytest.raises(TypeError):
        pv.Line(pointa, pointb, resolution=0.1)  # from vtk

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

    with pytest.raises(ValueError):  # noqa: PT011
        pv.MultipleLines(points[:, :1])

    with pytest.raises(ValueError):  # noqa: PT011
        pv.MultipleLines(points[0, :])


def test_tube():
    pointa = (0, 0, 0)
    pointb = (10, 1.0, 3)

    tube = pv.Tube(n_sides=3)
    assert tube.n_points == 6
    assert tube.n_cells == 3
    tube = pv.Tube(pointa=pointa, pointb=pointb, resolution=10)
    assert tube.n_points == 165
    assert tube.n_cells == 15

    with pytest.raises(ValueError):  # noqa: PT011
        pv.Tube(pointa=pointa, pointb=pointb, resolution=-1)

    with pytest.raises(TypeError):
        pv.Tube(pointa=pointa, pointb=pointb, resolution=0.1)  # from vtk

    with pytest.raises(TypeError):
        pv.Tube(pointa=(0, 0), pointb=pointb)

    with pytest.raises(TypeError):
        pv.Tube(pointa=pointa, pointb=(10, 1.0))


@pytest.mark.parametrize('capping', [True, False])
def test_tube_capping(capping: bool):
    # Clean due to duplicated points at the cylinder end borders.
    tube: pv.PolyData = pv.Tube(capping=capping).clean().triangulate()
    assert tube.is_manifold is capping


def test_capsule():
    capsule = pv.Capsule()
    assert np.any(capsule.points)
    assert np.any(capsule.faces)


# https://github.com/pyvista/pyvista/pull/6119
@pytest.mark.parametrize('center', [(4, 5, 6), (1, 1, 1)])
@pytest.mark.parametrize('direction', [(0, 1, -1), (1, 1, 0)])
def test_capsule_center(center, direction):
    capsule = pv.Capsule(center=center, direction=direction)
    cylinder = pv.Cylinder(center=center, direction=direction)
    assert np.allclose(capsule.center, cylinder.center)


def test_cube():
    cube = pv.Cube()
    assert np.any(cube.points)
    assert np.any(cube.faces)
    bounds = (1.0, 3.0, 5.0, 6.0, 7.0, 8.0)
    cube = pv.Cube(bounds=bounds)
    assert np.any(cube.points)
    assert np.any(cube.faces)
    assert np.allclose(cube.bounds, bounds)

    assert 'Normals' in cube.point_data.keys()
    normals = cube.point_data['Normals']
    expected = 0.57735026
    assert np.allclose(np.abs(normals), expected)


@pytest.mark.parametrize(('point_dtype'), (['float32', 'float64', 'invalid']))
def test_cube_point_dtype(point_dtype):
    if point_dtype in ['float32', 'float64']:
        cube = pv.Cube(point_dtype=point_dtype)
        assert cube.points.dtype == point_dtype
    else:
        with pytest.raises(ValueError, match="Point dtype must be either 'float32' or 'float64'"):
            _ = pv.Cube(point_dtype=point_dtype)


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
    mesh1 = pv.Box(bounds, level=level, quads=quads)
    assert mesh1.n_cells == (level + 1) * (level + 1) * 6
    assert np.allclose(mesh1.bounds, bounds)

    quads = False
    mesh2 = pv.Box(bounds, level=level, quads=quads)
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
    mesh = pv.Text3D('foo', depth=0.5, width=2, height=3, normal=(0, 0, 1), center=(1, 2, 3))
    assert mesh.n_points
    assert mesh.n_cells

    actual_width, actual_height, actual_depth = mesh.bounds_size
    assert np.isclose(actual_width, 2.0)
    assert np.isclose(actual_height, 3.0)
    assert np.isclose(actual_depth, 0.5)
    assert np.allclose(mesh.center, [1.0, 2.0, 3.0])

    # Test setting empty string returns empty mesh with zeros as bounds
    mesh = pv.Text3D(string='')
    assert mesh.n_points == 1
    assert mesh.bounds == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_wavelet():
    mesh = pv.Wavelet()
    assert mesh.n_points
    assert mesh.n_cells


def test_circular_arc():
    pointa = [-1, 0, 0]
    pointb = [0, 1, 0]
    center = [0, 0, 0]
    resolution = 100

    mesh = pv.CircularArc(pointa=pointa, pointb=pointb, center=center, resolution=resolution)
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01) * np.pi / 2.0
    assert np.allclose(mesh['Distance'], distance)

    # pointa and pointb are not equidistant from center
    with pytest.raises(ValueError):  # noqa: PT011
        mesh = pv.CircularArc(
            pointa=[-1, 0, 0], pointb=[-0.99, 0.001, 0], center=[0, 0, 0], resolution=100
        )


def test_circular_arc_from_normal():
    center = [0, 0, 0]
    normal = [0, 0, 1]
    polar = [-2.0, 0, 0]
    angle = 90
    resolution = 100

    mesh = pv.CircularArcFromNormal(
        center=center, resolution=resolution, normal=normal, polar=polar, angle=angle
    )
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


@pytest.mark.parametrize(
    'points',
    [
        ([3.0, 1.0, 1.0], [3.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]),
        (
            [0.043, 0.0359, 0.0001],
            [0.044, 0.0359, 0.0001],
            [0.043, 0.036, 0.0001],
            [0.044, 0.036, 0.0001],
        ),
    ],
)
def test_rectangle(points):
    pointa, pointb, pointc, pointd = points

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pv.core.utilities.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pv.core.utilities.transformations.apply_transformation_to_points(
        trans,
        np.array([pointa, pointb, pointc, pointd]),
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
        trans,
        np.array([pointa, pointb, pointc]),
    )

    with pytest.raises(ValueError, match='The three points should defined orthogonal vectors'):
        pv.Rectangle(rotated)


def test_rectangle_two_identical_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]
    pointc = [3.0, 1.0, 1.0]

    # Do a rotation to be in full 3D space with floating point coordinates
    trans = pv.core.utilities.transformations.axis_angle_rotation([1, 1, 1], 30)
    rotated = pv.core.utilities.transformations.apply_transformation_to_points(
        trans,
        np.array([pointa, pointb, pointc]),
    )

    with pytest.raises(
        ValueError,
        match='Unable to build a rectangle with less than three different points',
    ):
        pv.Rectangle(rotated)


def test_rectangle_not_enough_points():
    pointa = [3.0, 1.0, 1.0]
    pointb = [4.0, 3.0, 1.0]

    with pytest.raises(TypeError, match='Points must be given as length 3 np.ndarray or list'):
        pv.Rectangle([pointa, pointb])


def test_circle():
    radius = 1.0

    mesh = pv.Circle(radius=radius)
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
    ('kind_str', 'kind_int', 'n_vertices', 'n_faces'),
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
    assert solid_from_str.n_faces_strict == n_faces


def test_platonic_invalids():
    with pytest.raises(ValueError):  # noqa: PT011
        pv.PlatonicSolid(kind='invalid')
    with pytest.raises(ValueError):  # noqa: PT011
        pv.PlatonicSolid(kind=42)
    with pytest.raises(ValueError):  # noqa: PT011
        pv.PlatonicSolid(kind=[])


def test_tetrahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Tetrahedron(radius=radius, center=center)
    assert solid.n_points == 4
    assert solid.n_faces_strict == 4
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Tetrahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_octahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Octahedron(radius=radius, center=center)
    assert solid.n_points == 6
    assert solid.n_faces_strict == 8
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Octahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_dodecahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Dodecahedron(radius=radius, center=center)
    assert solid.n_points == 20
    assert solid.n_faces_strict == 12
    assert np.allclose(solid.center, center)

    doubled_solid = pv.Dodecahedron(radius=2 * radius, center=center)
    assert np.isclose(doubled_solid.length, 2 * solid.length)


def test_icosahedron():
    radius = 1.7
    center = (1, -2, 3)
    solid = pv.Icosahedron(radius=radius, center=center)
    assert solid.n_points == 12
    assert solid.n_faces_strict == 20
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
    assert icosahedron.n_faces_strict * 4**nsub == icosphere.n_faces_strict
