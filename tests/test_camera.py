import sys

import numpy as np
import pytest

import pyvista


def test_camera():

    camera = pyvista.Camera()

    position = np.random.random(3)
    camera.position = position
    assert np.all(camera.GetPosition() == position)
    assert np.all(camera.position == position)

    focal_point = np.random.random(3)
    camera.focal_point = focal_point
    assert np.all(camera.GetFocalPoint() == focal_point)
    assert np.all(camera.focal_point == focal_point)

    model_transform_matrix = np.random.random((4, 4))
    camera.model_transform_matrix = model_transform_matrix
    assert np.all(camera.model_transform_matrix == model_transform_matrix)

    camera.distance == np.linalg.norm(focal_point - position, ord=2)

    thickness = np.random.random(1)
    camera.thickness = thickness
    assert camera.thickness == thickness

    parallel_scale = np.random.random(1)
    camera.parallel_scale = parallel_scale
    assert camera.parallel_scale == parallel_scale

    value = np.random.random(1)
    camera.zoom(value)

    vector = np.random.random(3)
    camera.up(vector)
    camera.up()

    camera.enable_parallel_projection()
    assert camera.GetParallelProjection()
    assert camera.is_parallel_projection

    camera.disable_parallel_projection()
    assert not camera.GetParallelProjection()
    assert not camera.is_parallel_projection

    near_point = np.random.random(1)
    far_point = near_point + np.random.random(1)
    points = (near_point, far_point)
    camera.clipping_range = points
    assert camera.GetClippingRange() == points
    assert camera.clipping_range == points

    with pytest.raises(ValueError):
        far_point = near_point - np.random.random(1)
        points = (near_point, far_point)
        camera.clipping_range = points

