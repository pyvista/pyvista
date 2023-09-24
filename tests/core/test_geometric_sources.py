import numpy as np
import pytest

import pyvista as pv


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


def test_line_source():
    algo = pv.LineSource()
    assert algo.resolution == 1
    assert np.array_equal(algo.points == [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    points = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    algo = pv.LineSource(points=points)
    assert np.array_equal(algo.points == points)
