import numpy as np
import pytest

import pyvista
from pyvista.utilities.misc import PyvistaDeprecationWarning


@pytest.fixture()
def camera():
    return pyvista.Camera()


def test_invalid_init():
    with pytest.raises(TypeError):
        pyvista.Camera(1)


def test_camera_position(camera):
    position = np.random.random(3)
    camera.position = position
    assert np.all(camera.GetPosition() == position)
    assert np.all(camera.position == position)


def test_focal_point(camera):
    focal_point = np.random.random(3)
    camera.focal_point = focal_point
    assert np.all(camera.GetFocalPoint() == focal_point)
    assert np.all(camera.focal_point == focal_point)


def test_model_transform_matrix(camera):
    model_transform_matrix = np.random.random((4, 4))
    camera.model_transform_matrix = model_transform_matrix
    assert np.all(camera.model_transform_matrix == model_transform_matrix)


def test_distance(camera):
    focal_point = np.random.random(3)
    position = np.random.random(3)
    camera.position = position
    camera.focal_point = focal_point
    assert np.isclose(camera.distance, np.linalg.norm(focal_point - position, ord=2),
                      rtol=1E-8)


def test_thickness(camera):
    thickness = np.random.random(1)
    camera.thickness = thickness
    assert camera.thickness == thickness


def test_parallel_scale(camera):
    parallel_scale = np.random.random(1)
    camera.parallel_scale = parallel_scale
    assert camera.parallel_scale == parallel_scale


def test_zoom(camera):
    value = np.random.random(1)
    camera.enable_parallel_projection()
    orig_scale = camera.parallel_scale
    zoom = 0.5
    camera.zoom(zoom)
    assert camera.parallel_scale == orig_scale/zoom


def test_up(camera):
    up = (0.410018, 0.217989, 0.885644)
    camera.up = up
    assert np.allclose(camera.up, up)


def test_enable_parallel_projection(camera):
    camera.enable_parallel_projection()
    assert camera.GetParallelProjection()
    assert camera.parallel_projection


def test_disable_parallel_projection(camera):
    camera.disable_parallel_projection()
    assert not camera.GetParallelProjection()
    assert not camera.parallel_projection


def test_clipping_range(camera):
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


def test_reset_clipping_range(camera):
    with pytest.raises(AttributeError):
        camera.reset_clipping_range()

    # requires renderer for this method
    crng = (1, 2)
    pl = pyvista.Plotter()
    pl.add_mesh(pyvista.Sphere())
    pl.camera.clipping_range = crng
    assert pl.camera.clipping_range == crng
    pl.camera.reset_clipping_range()
    assert pl.camera.clipping_range != crng


def test_view_angle(camera):
    assert camera.GetViewAngle() == camera.view_angle


def test_direction(camera):
    assert camera.GetDirectionOfProjection() == camera.direction


def test_view_frustum(camera):
    frustum = camera.view_frustum(1.0)
    assert frustum.n_points == 8
    assert frustum.n_cells == 6


def test_roll(camera):
    angle = 360.0*(np.random.rand()-0.5)
    camera.roll = angle
    assert np.allclose(camera.GetRoll(), angle)
    assert np.allclose(camera.roll, angle)


def test_elevation(camera):
    position = (1.0, 0.0, 0.0)
    elevation = 90.0
    camera.up = (0.0, 0.0, 1.0)
    camera.position = position
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.elevation = elevation
    assert np.allclose(camera.position, (0.0, 0.0, 1.0))
    assert np.allclose(camera.GetPosition(), (0.0, 0.0, 1.0))
    assert np.allclose(camera.elevation, elevation)

    camera.position = (2.0, 0.0, 0.0)
    assert np.allclose(camera.GetPosition(), (2.0, 0.0, 0.0))

    camera.elevation = 180.0
    assert np.allclose(camera.GetPosition(), (-2.0, 0.0, 0.0))


def test_azimuth(camera):
    position = (1.0, 0.0, 0.0)
    azimuth = 90.0
    camera.up = (0.0, 0.0, 1.0)
    camera.position = position
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.azimuth = azimuth
    assert np.allclose(camera.position, (0.0, 1.0, 0.0))
    assert np.allclose(camera.GetPosition(), (0.0, 1.0, 0.0))
    assert np.allclose(camera.azimuth, azimuth)

    camera.position = (2.0, 0.0, 0.0)
    assert np.allclose(camera.GetPosition(), (2.0, 0.0, 0.0))

    camera.azimuth = 180.0
    assert np.allclose(camera.GetPosition(), (-2.0, 0.0, 0.0))

def test_deprecation_warning_of_is_parallel_projection(camera):
    with pytest.warns(PyvistaDeprecationWarning):
        _ = camera.is_parallel_projection
