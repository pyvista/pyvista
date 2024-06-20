from __future__ import annotations

import re

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.geometric_objects import translate


@pytest.fixture()
def axes_geometry_source():
    return pv.AxesGeometrySource()


def test_capsule_source():
    if pv.vtk_version_info < (9, 3):
        algo = pv.CapsuleSource()
        assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
        assert np.array_equal(algo.direction, (1.0, 0.0, 0.0))
        assert algo.radius == 0.5
        assert algo.cylinder_length == 1.0
        assert algo.theta_resolution == 30
        assert algo.phi_resolution == 30
        direction = np.random.default_rng().random(3)
        algo.direction = direction
        assert np.array_equal(algo.direction, direction)


def test_cone_source():
    algo = pv.ConeSource()
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert np.array_equal(algo.direction, (1.0, 0.0, 0.0))
    assert algo.height == 1.0
    assert algo.radius == 0.5
    assert algo.capping
    assert algo.resolution == 6
    with pytest.raises(ValueError):  # noqa: PT011
        algo = pv.ConeSource(angle=0.0, radius=0.0)
    algo = pv.ConeSource(angle=0.0)
    assert algo.angle == 0.0
    algo = pv.ConeSource(radius=0.0)
    assert algo.radius == 0.0


def test_cylinder_source():
    algo = pv.CylinderSource()
    assert algo.radius == 0.5
    assert algo.height == 1.0
    assert algo.capping
    assert algo.resolution == 100
    center = np.random.default_rng().random(3)
    direction = np.random.default_rng().random(3)
    algo.center = center
    algo.direction = direction
    assert np.array_equal(algo.center, center)
    assert np.array_equal(algo.direction, direction)


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_cylinder_capsule_cap():
    algo = pv.CylinderSource()
    algo.capsule_cap = True
    assert algo.capsule_cap


def test_multiple_lines_source():
    algo = pv.MultipleLinesSource()
    points = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert np.array_equal(algo.points, points)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    algo = pv.MultipleLinesSource(points=points)
    assert np.array_equal(algo.points, points)
    with pytest.raises(ValueError, match='Array of points must have three values per point'):
        algo.points = points[:, :1]
    with pytest.raises(ValueError, match='>=2 points need to define multiple lines.'):
        algo.points = points[0, :]


@pytest.fixture()
def bunny():
    return examples.download_bunny_coarse()


@pytest.mark.parametrize("is_negative", [True, False])
@pytest.mark.parametrize("delta", [([0, 0, 0]), ([1e-8, 0, 0]), ([0, 0, 1e-8])])
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


def test_translate_precision():
    """
    Test that specifying a 64bit float as an arg, will not
    introduce precision error for 32bit meshes.
    """
    val = np.float64(29380 / 18)

    # test indirectly using Plane, which will yield a float32 mesh
    mesh = pv.Plane(center=[0, val / 2, 0], j_size=val, i_resolution=1, j_resolution=1)
    assert mesh.points.dtype == np.float32

    expected = np.array(
        [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 1632.2222, 0.0], [0.5, 1632.2222, 0.0]],
        dtype=np.float32,
    )

    # DO NOT USE np.all_close. This should match exactly
    assert np.array_equal(mesh.points, expected)


def test_text3d_source_empty_string():
    # Test empty string is processed to have a single point
    src = pv.Text3DSource(process_empty_string=True)
    assert src.process_empty_string
    out = src.output
    assert out.n_points == 1

    # Test setting an empty string is processed to have a single point
    src.process_empty_string = False
    assert not src.process_empty_string
    out = src.output
    assert out.n_points == 0

    if pv.vtk_version_info == (9, 0, 3):
        mx, mn = 1, -1
    else:
        mx, mn = vtk.VTK_DOUBLE_MAX, vtk.VTK_DOUBLE_MIN

    assert out.bounds == (mx, mn, mx, mn, mx, mn)


def test_text3d_source():
    src = pv.Text3DSource(string='Text')
    assert src.string == 'Text'
    out = src.output
    assert len(out.split_bodies()) == 4


@pytest.mark.parametrize('string', [" ", 'TEXT'])
@pytest.mark.parametrize('center', [(0, 0, 0), (1, -2, 3)])
@pytest.mark.parametrize('height', [None, 0, 1, 2])
@pytest.mark.parametrize('width', [None, 0, 1, 2])
@pytest.mark.parametrize('depth', [None, 0, 1, 2])
@pytest.mark.parametrize('normal', [(0, 0, 1)])
def test_text3d_source_parameters(string, center, height, width, depth, normal):
    src = pv.Text3DSource(
        string=string,
        center=center,
        height=height,
        width=width,
        depth=depth,
        normal=normal,
    )
    out = src.output
    bnds = out.bounds
    actual_width, actual_height, actual_depth = (
        bnds[1] - bnds[0],
        bnds[3] - bnds[2],
        bnds[5] - bnds[4],
    )

    # Compute expected values
    empty_string = string.isspace()
    if empty_string:
        expected_width, expected_height, expected_depth = 0.0, 0.0, 0.0
    else:
        expected_width, expected_height, expected_depth = width, height, depth

        # Determine expected values for cases where width, height, or depth is None
        if depth is None:
            expected_depth = actual_height * 0.5

        # For width and height, create an unscaled version for reference
        src_not_scaled = pv.Text3DSource(string=string, center=center)
        out_not_scaled = src_not_scaled.output
        bnds = out_not_scaled.bounds
        unscaled_width, unscaled_height = bnds[1] - bnds[0], bnds[3] - bnds[2]
        if width is None and height is not None:
            expected_width = unscaled_width * actual_height / unscaled_height
        elif width is not None and height is None:
            expected_height = unscaled_height * actual_width / unscaled_width
        elif width is None and height is None:
            expected_width = unscaled_width
            expected_height = unscaled_height

    assert np.allclose(actual_width, expected_width)
    assert np.allclose(actual_height, expected_height)
    assert np.allclose(actual_depth, expected_depth)
    assert np.array_equal(out.center, center)

    if not empty_string and width and height and depth == 0:
        # Test normal direction for planar 2D meshes
        actual_normal = np.mean(out.cell_normals, axis=0)
        assert np.allclose(actual_normal, normal)

        # Since `direction` param is under-determined and may swap the
        # width and height, test normal again without testing the bounds
        # We also use a symmetric text string since the oriented mesh's
        # bounding box center and/or the mean of its points will otherwise
        # vary and is challenging to test
        new_normal = np.array((1, -2, 3))
        src = pv.Text3DSource(string='I', center=center, normal=new_normal, depth=0)
        out = src.output
        actual_normal = np.mean(out.cell_normals, axis=0)
        expected_normal = new_normal / np.linalg.norm(new_normal)
        assert np.allclose(actual_normal, expected_normal, atol=1e-4)

        points_center = np.mean(out.points, axis=0)
        assert np.allclose(points_center, center, atol=1e-4)


@pytest.fixture()
def text3d_source_with_text():
    return pv.Text3DSource("TEXT")


def test_text3d_source_update(text3d_source_with_text):
    assert text3d_source_with_text._modified
    assert text3d_source_with_text._output.n_points == 0
    text3d_source_with_text.update()
    assert not text3d_source_with_text._modified
    assert text3d_source_with_text._output.n_points > 1

    # Test calling update has no effect on output when modified flag is not set
    points_before = text3d_source_with_text._output.GetPoints()
    text3d_source_with_text.update()
    points_after = text3d_source_with_text._output.GetPoints()
    assert not text3d_source_with_text._modified
    assert points_before is points_after


def text3d_source_test_params():
    return (
        ('string', 'TEXT'),
        ('center', (1, 2, 3)),
        ('normal', (4, 5, 6)),
        ('height', 2),
        ('width', 3),
        ('depth', 4),
    )


def test_text3d_source_output(text3d_source_with_text):
    # Store initial object references
    out1 = text3d_source_with_text._output
    out1_points = out1.GetPoints()
    assert out1.n_points == 0

    # Test getting output triggers an update
    assert text3d_source_with_text._modified
    out2 = text3d_source_with_text.output
    assert not text3d_source_with_text._modified

    # Test that output object reference is unchanged
    assert out2 is out1

    # Test that output points object reference is changed
    out2_points = out2.GetPoints()
    assert out2_points is not out1_points

    # Test correct output
    assert len(out2.split_bodies()) == len(text3d_source_with_text.string)


@pytest.mark.parametrize(
    'kwarg_tuple',
    text3d_source_test_params(),
)
def test_text3d_source_modified_init(kwarg_tuple):
    # Test init modifies source but does not update output
    name, value = kwarg_tuple
    kwarg_dict = {name: value}

    src = pv.Text3DSource(**kwarg_dict)
    assert src._modified
    assert src._output.n_points == 0


@pytest.mark.parametrize(
    'kwarg_tuple',
    text3d_source_test_params(),
)
def test_text3d_source_modified(text3d_source_with_text, kwarg_tuple):
    # Set test param
    name, value = kwarg_tuple
    setattr(text3d_source_with_text, name, value)
    assert text3d_source_with_text._modified

    # Call update to clear modified flag
    assert text3d_source_with_text._modified
    text3d_source_with_text.update()
    assert not text3d_source_with_text._modified

    # Test that setting the same value does not set the modified flag
    points_before = text3d_source_with_text._output.GetPoints()  # Manually set flag for test
    setattr(text3d_source_with_text, name, value)
    points_after = text3d_source_with_text._output.GetPoints()
    assert not text3d_source_with_text._modified
    assert points_before is points_after

    # Test setting a new value sets modified flag but does not change output
    new_value = value + value if name == 'string' else np.array(value) * 2
    points_before = text3d_source_with_text._output.GetPoints()
    setattr(text3d_source_with_text, name, new_value)
    points_after = text3d_source_with_text._output.GetPoints()
    assert text3d_source_with_text._modified
    assert points_before is points_after


def test_disc_source():
    algo = pv.DiscSource()
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert algo.inner == 0.25
    assert algo.outer == 0.5
    assert algo.r_res == 1
    assert algo.c_res == 6
    if pv.vtk_version_info >= (9, 2):
        center = (1.0, 2.0, 3.0)
        algo = pv.DiscSource(center=center)
        assert algo.center == center


def test_cube_source():
    algo = pv.CubeSource()
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert algo.x_length == 1.0
    assert algo.y_length == 1.0
    assert algo.z_length == 1.0
    bounds = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    algo = pv.CubeSource(bounds=bounds)
    assert np.array_equal(algo.bounds, bounds)
    with pytest.raises(TypeError):
        algo = pv.CubeSource(bounds=0.0)


def test_sphere_source():
    algo = pv.SphereSource()
    assert algo.radius == 0.5
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert algo.theta_resolution == 30
    assert algo.phi_resolution == 30
    assert algo.start_theta == 0.0
    assert algo.end_theta == 360.0
    assert algo.start_phi == 0.0
    assert algo.end_phi == 180.0
    center = (1.0, 2.0, 3.0)
    if pv.vtk_version_info >= (9, 2):
        algo = pv.SphereSource(center=center)
        assert algo.center == center


def test_line_source():
    algo = pv.LineSource()
    assert np.array_equal(algo.pointa, (-0.5, 0.0, 0.0))
    assert np.array_equal(algo.pointb, (0.5, 0.0, 0.0))
    assert algo.resolution == 1


def test_polygon_source():
    algo = pv.PolygonSource()
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert algo.radius == 1.0
    assert np.array_equal(algo.normal, (0.0, 0.0, 1.0))
    assert algo.n_sides == 6
    assert algo.fill


def test_platonic_solid_source():
    algo = pv.PlatonicSolidSource()
    assert algo.kind == 'tetrahedron'


def test_plane_source():
    algo = pv.PlaneSource()
    assert algo.i_resolution == 10
    assert algo.j_resolution == 10


def test_superquadric_source():
    algo = pv.SuperquadricSource()
    assert algo.center == (0.0, 0.0, 0.0)
    assert algo.scale == (1.0, 1.0, 1.0)
    assert algo.size == 0.5
    assert algo.theta_roundness == 1.0
    assert algo.phi_roundness == 1.0
    assert algo.theta_resolution == 16
    assert algo.phi_resolution == 16
    assert not algo.toroidal
    assert algo.thickness == 1 / 3


def test_arrow_source():
    algo = pv.ArrowSource()
    assert algo.tip_length == 0.25
    assert algo.tip_radius == 0.1
    assert algo.tip_resolution == 20
    assert algo.shaft_radius == 0.05
    assert algo.shaft_resolution == 20


def test_box_source():
    algo = pv.BoxSource()
    assert np.array_equal(algo.bounds, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    assert algo.level == 0
    assert algo.quads


def test_axes_geometry_source_symmetric_set_get(axes_geometry_source):
    assert axes_geometry_source.symmetric is False
    axes_geometry_source.symmetric = True
    assert axes_geometry_source.symmetric is True


def test_axes_geometry_source_symmetric_init():
    axes_geometry_source = pv.AxesGeometrySource(symmetric=True)
    assert axes_geometry_source.output.bounds == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_axes_geometry_source_total_length_set_get(axes_geometry_source):
    assert axes_geometry_source.total_length == (1, 1, 1)
    new_length = (1.1, 2.2, 3.3)
    axes_geometry_source.total_length = new_length
    assert axes_geometry_source.total_length == new_length

    match = 'Total length (0.1, 0.1, 0.1) cannot be less than the tip length (0.2, 0.2, 0.2) when normalized mode is disabled.'
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_geometry_source.total_length = 0.1


def test_axes_geometry_source_total_length_init():
    axes_geometry_source = pv.AxesGeometrySource(total_length=2)
    assert axes_geometry_source.total_length == (2, 2, 2)


def test_axes_geometry_source_total_length_bounds(axes_geometry_source):
    x_len, y_len, z_len = 1.5, 2.5, 3.5
    axes_geometry_source.total_length = x_len, y_len, z_len
    actual_bounds = axes_geometry_source.output.bounds

    tip_radius = 0.1  # default value
    expected_bounds = (-tip_radius, x_len, -tip_radius, y_len, -tip_radius, z_len)
    assert np.allclose(actual_bounds, expected_bounds)


def test_axes_geometry_source_shaft_length_set_get(axes_geometry_source):
    assert axes_geometry_source.shaft_length == (0.8, 0.8, 0.8)
    new_length = (0.1, 0.2, 0.3)
    axes_geometry_source.shaft_length = new_length
    assert axes_geometry_source.shaft_length == new_length


def test_axes_geometry_source_shaft_length_init():
    axes_geometry_source = pv.AxesGeometrySource(shaft_length=0.9)
    assert axes_geometry_source.shaft_length == (0.9, 0.9, 0.9)


@pytest.mark.parametrize('part', ['x_shaft', 'y_shaft', 'z_shaft', 'x_tip', 'y_tip', 'z_tip'])
def test_axes_geometry_source_bounds(axes_geometry_source, part):
    x_shaft_len, y_shaft_len, z_shaft_len = 0.5, 0.6, 0.7
    shaft_radius = 0.05
    axes_geometry_source.shaft_length = x_shaft_len, y_shaft_len, z_shaft_len
    axes_geometry_source.shaft_radius = shaft_radius

    x_tip_len, y_tip_len, z_tip_len = 0.2, 0.3, 0.4
    tip_radius = 0.2
    axes_geometry_source.tip_length = x_tip_len, y_tip_len, z_tip_len
    axes_geometry_source.tip_radius = tip_radius

    x_shaft, y_shaft, z_shaft, x_tip, y_tip, z_tip = axes_geometry_source.output

    if part == 'x_shaft':
        actual_bounds = x_shaft.bounds
        expected_bounds = (
            0,
            x_shaft_len,
            -shaft_radius,
            shaft_radius,
            -shaft_radius,
            shaft_radius,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    elif part == 'y_shaft':
        actual_bounds = y_shaft.bounds
        expected_bounds = (
            -shaft_radius,
            shaft_radius,
            0,
            y_shaft_len,
            -shaft_radius,
            shaft_radius,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    elif part == 'z_shaft':
        actual_bounds = z_shaft.bounds
        expected_bounds = (
            -shaft_radius,
            shaft_radius,
            -shaft_radius,
            shaft_radius,
            0,
            z_shaft_len,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    elif part == 'x_tip':
        actual_bounds = x_tip.bounds
        expected_bounds = (
            x_shaft_len,
            x_shaft_len + x_tip_len,
            -tip_radius,
            tip_radius,
            -tip_radius,
            tip_radius,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    elif part == 'y_tip':
        actual_bounds = y_tip.bounds
        expected_bounds = (
            -tip_radius,
            tip_radius,
            y_shaft_len,
            y_shaft_len + y_tip_len,
            -tip_radius,
            tip_radius,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    elif part == 'z_tip':
        actual_bounds = z_tip.bounds
        expected_bounds = (
            -tip_radius,
            tip_radius,
            -tip_radius,
            tip_radius,
            z_shaft_len,
            z_shaft_len + z_tip_len,
        )
        assert np.allclose(actual_bounds, expected_bounds)

    else:
        raise NotImplementedError


def test_axes_geometry_source_tip_length_set_get(axes_geometry_source):
    assert axes_geometry_source.tip_length == (0.2, 0.2, 0.2)
    axes_geometry_source.tip_length = (0.1, 0.2, 0.3)
    assert axes_geometry_source.tip_length == (0.1, 0.2, 0.3)


def test_axes_geometry_source_tip_length_init():
    axes_geometry_source = pv.AxesGeometrySource(tip_length=0.9)
    assert axes_geometry_source.tip_length == (0.9, 0.9, 0.9)


def test_axes_geometry_source_tip_radius_set_get(axes_geometry_source):
    assert axes_geometry_source.tip_radius == 0.1
    axes_geometry_source.tip_radius = 0.8
    assert axes_geometry_source.tip_radius == 0.8


def test_axes_geometry_source_tip_radius_init():
    axes_geometry_source = pv.AxesGeometrySource(tip_radius=9)
    assert axes_geometry_source.tip_radius == 9


@pytest.mark.parametrize(
    'shaft_type',
    pv.AxesGeometrySource.GEOMETRY_TYPES,
)
def test_axes_geometry_source_shaft_type_set_get(shaft_type, axes_geometry_source):
    axes_geometry_source.shaft_type = shaft_type
    assert axes_geometry_source.shaft_type == shaft_type


@pytest.mark.parametrize(
    'shaft_type',
    pv.AxesGeometrySource.GEOMETRY_TYPES,
)
def test_axes_geometry_source_shaft_type_init(shaft_type):
    axes_geometry_source = pv.AxesGeometrySource(shaft_type=shaft_type)
    assert axes_geometry_source.shaft_type == shaft_type


@pytest.mark.parametrize(
    'tip_type',
    pv.AxesGeometrySource.GEOMETRY_TYPES,
)
def test_axes_geometry_source_tip_type_set_get(tip_type, axes_geometry_source):
    axes_geometry_source.tip_type = tip_type
    assert axes_geometry_source.tip_type == tip_type


@pytest.mark.parametrize(
    'tip_type',
    pv.AxesGeometrySource.GEOMETRY_TYPES,
)
def test_axes_geometry_source_tip_type_init(tip_type):
    axes_geometry_source = pv.AxesGeometrySource(tip_type=tip_type)
    assert axes_geometry_source.tip_type == tip_type


def test_axes_geometry_source_axis_color_set_get(axes_geometry_source):
    assert axes_geometry_source.x_color[0].name == 'tomato'
    assert axes_geometry_source.x_color[1].name == 'tomato'
    assert axes_geometry_source.y_color[0].name == 'seagreen'
    assert axes_geometry_source.y_color[1].name == 'seagreen'
    assert axes_geometry_source.z_color[0].name == 'blue'
    assert axes_geometry_source.z_color[1].name == 'blue'

    axes_geometry_source.x_color = 'purple'
    assert len(axes_geometry_source.x_color) == 2
    assert axes_geometry_source.x_color[0].name == 'purple'
    assert axes_geometry_source.x_color[1].name == 'purple'

    axes_geometry_source.x_color = [1, 2, 3]
    assert np.array_equal(axes_geometry_source.x_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_geometry_source.x_color[1].int_rgb, [1, 2, 3])

    axes_geometry_source.y_color = 'purple'
    assert axes_geometry_source.y_color[0].name == 'purple'
    assert axes_geometry_source.y_color[1].name == 'purple'
    axes_geometry_source.y_color = [1, 2, 3]
    assert np.array_equal(axes_geometry_source.y_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_geometry_source.y_color[1].int_rgb, [1, 2, 3])

    axes_geometry_source.z_color = 'purple'
    assert axes_geometry_source.z_color[0].name == 'purple'
    assert axes_geometry_source.z_color[1].name == 'purple'
    axes_geometry_source.z_color = [1, 2, 3]
    assert np.array_equal(axes_geometry_source.z_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_geometry_source.z_color[1].int_rgb, [1, 2, 3])


def test_axes_geometry_source_axis_color_init():
    axes_geometry_source = pv.AxesGeometrySource(
        x_color='yellow',
        y_color='orange',
        z_color='purple',
    )
    assert axes_geometry_source.x_color[0].name == 'yellow'
    assert axes_geometry_source.x_color[1].name == 'yellow'
    assert axes_geometry_source.y_color[0].name == 'orange'
    assert axes_geometry_source.y_color[1].name == 'orange'
    assert axes_geometry_source.z_color[0].name == 'purple'
    assert axes_geometry_source.z_color[1].name == 'purple'


def test_axes_geometry_source_shaft_radius_set_get(axes_geometry_source):
    assert axes_geometry_source.shaft_radius == 0.025
    axes_geometry_source.shaft_radius = 0.1
    assert axes_geometry_source.shaft_radius == 0.1


def test_axes_geometry_source_shaft_radius_init():
    axes_geometry_source = pv.AxesGeometrySource(shaft_radius=3)
    assert axes_geometry_source.shaft_radius == 3


@pytest.mark.parametrize(
    'test_prop_other_prop',
    [('shaft_length', 'tip_length'), ('tip_length', 'shaft_length')],
)
@pytest.mark.parametrize('decimals', list(range(8)))
@pytest.mark.parametrize('normalized_mode', [True, False])
def test_axes_geometry_source_normalized_mode(test_prop_other_prop, decimals, normalized_mode):
    test_prop, other_prop = test_prop_other_prop

    # Get default value
    axes_geometry_source = pv.AxesGeometrySource()
    other_prop_default = np.array(getattr(axes_geometry_source, other_prop))

    # Initialize actor with a random length for test prop
    random_length = np.round(np.random.default_rng().random(), decimals=decimals)
    var_kwargs = {}
    var_kwargs[test_prop] = random_length
    axes_geometry_source = pv.AxesGeometrySource(normalized_mode=normalized_mode, **var_kwargs)

    actual_test_prop = np.array(getattr(axes_geometry_source, test_prop))
    actual_other_prop = np.array(getattr(axes_geometry_source, other_prop))

    expected = np.array([random_length] * 3)
    assert np.array_equal(actual_test_prop, expected)
    if normalized_mode:
        # Test lengths sum to 1
        actual = actual_test_prop + actual_other_prop
        expected = (1, 1, 1)
        assert np.array_equal(actual, expected)
    else:
        # Test other length is unchanged
        actual = actual_other_prop
        expected = other_prop_default
        assert np.array_equal(actual, expected)

    # Test setting both does not raise error if they sum to one
    _ = pv.AxesGeometrySource(shaft_length=random_length, tip_length=1 - random_length)
    _ = pv.AxesGeometrySource(shaft_length=1 - random_length, tip_length=random_length)

    # test enabling auto_length after object has been created
    axes_geometry_source.normalized_mode = True
    setattr(axes_geometry_source, test_prop, 0.9)
    expected = (0.9, 0.9, 0.9)
    actual = getattr(axes_geometry_source, test_prop)
    assert np.array_equal(actual, expected)

    value_unchanged = (0.1, 0.1, 0.1)
    expected = value_unchanged
    actual = getattr(axes_geometry_source, other_prop)
    assert np.array_equal(actual, expected)

    # test disabling auto_length after object has been created
    axes_geometry_source.normalized_mode = False
    setattr(axes_geometry_source, test_prop, 0.7)
    expected = (0.7, 0.7, 0.7)
    actual = getattr(axes_geometry_source, test_prop)
    assert np.array_equal(actual, expected)

    expected = value_unchanged
    actual = getattr(axes_geometry_source, other_prop)
    assert np.array_equal(actual, expected)


def test_axes_geometry_source_normalized_mode_raises():
    ## normalized_mode=True
    # test no error when lengths sum to one
    pv.AxesGeometrySource(shaft_length=0.8, tip_length=0.2, normalized_mode=True)
    # test error raised otherwise
    match = (
        "Cannot set both `shaft_length` and `tip_length` with `normalized_mode` enabled'.\n"
        "Set either `shaft_length` or `tip_length`, but not both."
    )
    with pytest.raises(ValueError, match=match):
        pv.AxesGeometrySource(shaft_length=0.6, tip_length=0.6, normalized_mode=True)

    ## normalized_mode=False
    # test no error when lengths sum to total length
    pv.AxesGeometrySource(shaft_length=0.9, tip_length=0.2, total_length=1.1, normalized_mode=False)
    # test error raised otherwise
    match = (
        "Cannot set both `shaft_length` and `total_length` with `normalized_mode` disabled'.\n"
        "Set either `shaft_length` or `total_length`, but not both."
    )
    with pytest.raises(ValueError, match=match):
        pv.AxesGeometrySource(
            shaft_length=0.6,
            tip_length=0.2,
            total_length=1.1,
            normalized_mode=False,
        )


def test_axes_geometry_source_output():
    out = pv.AxesGeometrySource().output
    assert isinstance(out, pv.MultiBlock)
    assert out.keys() == ['x_shaft', 'y_shaft', 'z_shaft', 'x_tip', 'y_tip', 'z_tip']


def test_axes_geometry_source_rgb_scalars():
    out = pv.AxesGeometrySource(rgb_scalars=True).output
    assert all('axes_rgb' in block.array_names for block in out)

    out = pv.AxesGeometrySource(rgb_scalars=False).output
    assert all('axes_rgb' not in block.array_names for block in out)


def test_axes_geometry_source_repr(axes_geometry_source):
    repr_ = repr(axes_geometry_source)
    actual_lines = repr_.splitlines()[1:]
    expected_lines = [
        "  Shaft type:                 'cylinder'",
        '  Shaft radius:               0.025',
        '  Shaft length:               (0.8, 0.8, 0.8)',
        "  Tip type:                   'cone'",
        '  Tip radius:                 0.1',
        '  Tip length:                 (0.2, 0.2, 0.2)',
        '  Total length:               (1.0, 1.0, 1.0)',
        '  Position:                   (0.0, 0.0, 0.0)',
        '  Direction vectors:          [[1. 0. 0.]',
        '                               [0. 1. 0.]',
        '                               [0. 0. 1.]]',
        '  Symmetric:                  False',
        '  Normalized mode:            False',
        '  RGB scalars:                True',
        "  X color:                    ('tomato', 'tomato')",
        "  Y color:                    ('seagreen', 'seagreen')",
        "  Z color:                    ('blue', 'blue')",
    ]
    assert len(actual_lines) == len(expected_lines)
    assert actual_lines == expected_lines

    axes_geometry_source.shaft_type = pv.ParametricTorus()
    repr_ = repr(axes_geometry_source)
    assert "Shaft type:                 'custom'" in repr_
