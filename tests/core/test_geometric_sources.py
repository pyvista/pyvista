import numpy as np
import pytest

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
    with pytest.raises(ValueError):
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
    center = np.random.rand(3)
    direction = np.random.rand(3)
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


def test_text3d_source_string():
    src = pv.Text3DSource(process_empty_string=True)
    out = src.output
    assert out.n_points == 1

    src = pv.Text3DSource(process_empty_string=False)
    out = src.output
    assert out.n_points == 0

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


@pytest.mark.parametrize(
    'kwarg',
    [
        ('string', 'TEXT'),
        ('center', (1, 2, 3)),
        ('normal', (4, 5, 6)),
        ('height', 2),
        ('width', 3),
        ('depth', 4),
    ],
)
def test_text3d_source_modified(kwarg):
    # Set up test param
    name, value = kwarg
    test_kwargs = {name: value}

    # Make sure string is always set to ensure non-empty output
    string_val = 'TEXT'
    test_kwargs.setdefault('string', string_val)

    # Test init modifies source but does not update output
    src = pv.Text3DSource(**test_kwargs)
    assert src._modified
    assert src._output.n_points == 0

    # Store initial object references
    out1 = src._output
    out1_points = out1.GetPoints()

    # Test getting output triggers an update and resets modified flag
    out2 = src.output
    out2_points = out2.GetPoints()
    assert not src._modified
    assert len(out1.split_bodies()) == len(string_val)
    assert out2 is out1  # same output obj
    assert out2_points is not out1_points  # new points array

    # Test setting the same value does not modify obj
    setattr(src, name, value)
    assert not src._modified
    assert src._output is out1
    assert src._output.GetPoints() is out2_points
    assert len(src._output.split_bodies()) == len(string_val)

    # Test calling update has no effect on output when modified flag is not set
    src.update()
    assert not src._modified
    assert src._output is out1
    assert src._output.GetPoints() is out2_points
    assert len(src._output.split_bodies()) == len(string_val)

    # Test setting to a new value sets flag but does not modify the output
    if name == 'string':
        new_value = value + value
    else:
        new_value = np.array(value) * 2
    assert not np.array_equal(new_value, value)

    setattr(src, name, new_value)
    assert src._modified
    assert src._output is out1
    assert src._output.GetPoints() is out2_points
    assert len(src._output.split_bodies()) == len(string_val)

    # Test calling update modifies the output
    src.update()
    assert not src._modified
    assert src._output is out1
    assert src._output.GetPoints() is not out2_points
    assert len(src._output.split_bodies()) == len(src.string)
