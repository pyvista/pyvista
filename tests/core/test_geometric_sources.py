import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.geometric_objects import translate


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
        string=string, center=center, height=height, width=width, depth=depth, normal=normal
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
    if name == "string":
        new_value = value + value
    else:
        new_value = np.array(value) * 2
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
