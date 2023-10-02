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
