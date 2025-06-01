from __future__ import annotations

import io

import numpy as np
import pytest

import pyvista as pv

# pyvista attr -- value -- vtk name triples:
configuration = [
    ('position', (1, 1, 1), 'SetPosition'),
    ('focal_point', (2, 2, 2), 'SetFocalPoint'),
    (
        'model_transform_matrix',
        np.arange(4 * 4).reshape(4, 4),
        'SetModelTransformMatrix',
    ),
    ('thickness', 1, 'SetThickness'),
    ('parallel_scale', 2, 'SetParallelScale'),
    ('up', (0, 0, 1), 'SetViewUp'),
    ('clipping_range', (4, 5), 'SetClippingRange'),
    ('view_angle', 90.0, 'SetViewAngle'),
    ('roll', 180.0, 'SetRoll'),
]


@pytest.fixture
def camera():
    return pv.Camera()


@pytest.fixture
def paraview_pvcc():
    """Fixture returning a paraview camera file with values of the position"""
    tmp = """
    <PVCameraConfiguration description="ParaView camera configuration" version="1.0">
      <Proxy group="views" type="RenderView" id="6395" servers="21">
        <Property name="CameraPosition" id="6395.CameraPosition" number_of_elements="3">
          <Element index="0" value="10.519087611966333"/>
          <Element index="1" value="40.74973775632195"/>
          <Element index="2" value="-20.24019652397463"/>
        </Property>
        <Property name="CameraFocalPoint" id="6395.CameraFocalPoint" number_of_elements="3">
          <Element index="0" value="15.335762892470676"/>
          <Element index="1" value="-26.960151717473682"/>
          <Element index="2" value="17.860905595181094"/>
        </Property>
        <Property name="CameraViewUp" id="6395.CameraViewUp" number_of_elements="3">
          <Element index="0" value="0.2191945908188539"/>
          <Element index="1" value="-0.4665856879512876"/>
          <Element index="2" value="-0.8568847805596613"/>
        </Property>
        <Property name="CenterOfRotation" id="6395.CenterOfRotation" number_of_elements="3">
          <Element index="0" value="15.039424359798431"/>
          <Element index="1" value="-7.047080755233765"/>
          <Element index="2" value="6.712674975395203"/>
        </Property>
        <Property name="RotationFactor" id="6395.RotationFactor" number_of_elements="1">
          <Element index="0" value="1"/>
        </Property>
        <Property name="CameraViewAngle" id="6395.CameraViewAngle" number_of_elements="1">
          <Element index="0" value="30"/>
        </Property>
        <Property name="CameraParallelScale" id="6395.CameraParallelScale" number_of_elements="1">
          <Element index="0" value="20.147235678333413"/>
        </Property>
        <Property name="CameraParallelProjection" id="6395.CameraParallelProjection" number_of_elements="1">
          <Element index="0" value="0"/>
          <Domain name="bool" id="6395.CameraParallelProjection.bool"/>
        </Property>
      </Proxy>
    </PVCameraConfiguration>"""  # noqa: E501
    position = [10.519087611966333, 40.74973775632195, -20.24019652397463]
    focal = [15.335762892470676, -26.960151717473682, 17.860905595181094]
    view_up = [0.2191945908188539, -0.4665856879512876, -0.8568847805596613]
    view_angle = 30
    parallel_scale = 20.147235678333413
    projection = False

    return (
        io.StringIO(tmp),
        position,
        focal,
        view_up,
        view_angle,
        parallel_scale,
        projection,
    )


def test_invalid_init():
    with pytest.raises(TypeError):
        pv.Camera(1)


def test_camera_from_paraview_pvcc(paraview_pvcc):
    camera = pv.Camera.from_paraview_pvcc(paraview_pvcc[0])
    assert camera.position == pytest.approx(paraview_pvcc[1])
    assert camera.focal_point == pytest.approx(paraview_pvcc[2])
    assert camera.up == pytest.approx(paraview_pvcc[3])
    assert camera.view_angle == paraview_pvcc[4]
    assert camera.parallel_scale == paraview_pvcc[-2]
    assert camera.parallel_projection == paraview_pvcc[-1]


def test_camera_to_paraview_pvcc(camera, tmp_path):
    fname = tmp_path / 'test.pvcc'
    camera.to_paraview_pvcc(fname)
    assert fname.exists()
    ocamera = pv.Camera.from_paraview_pvcc(fname)
    assert ocamera == camera


def test_camera_position(camera):
    position = np.random.default_rng().random(3)
    camera.position = position
    assert np.all(camera.GetPosition() == position)
    assert np.all(camera.position == position)


def test_focal_point(camera):
    focal_point = np.random.default_rng().random(3)
    camera.focal_point = focal_point
    assert np.all(camera.GetFocalPoint() == focal_point)
    assert np.all(camera.focal_point == focal_point)


def test_model_transform_matrix(camera):
    model_transform_matrix = np.random.default_rng().random((4, 4))
    camera.model_transform_matrix = model_transform_matrix
    assert np.all(camera.model_transform_matrix == model_transform_matrix)


def test_distance(camera):
    focal_point = np.random.default_rng().random(3)
    position = np.random.default_rng().random(3)
    camera.position = position
    camera.focal_point = focal_point
    assert np.isclose(camera.distance, np.linalg.norm(focal_point - position, ord=2), rtol=1e-8)
    distance = np.random.default_rng().random()
    camera.distance = distance
    assert np.isclose(camera.distance, distance, atol=0.0002)
    # large absolute tolerance because of
    # https://github.com/Kitware/VTK/blob/5f855ff8f1237cbb5e5fa55a5ace48149237006e/Rendering/Core/vtkCamera.cxx#L563-L577


def test_thickness(camera):
    thickness = np.random.default_rng().random()
    camera.thickness = thickness
    assert camera.thickness == thickness


def test_parallel_scale(camera):
    parallel_scale = np.random.default_rng().random()
    camera.parallel_scale = parallel_scale
    assert camera.parallel_scale == parallel_scale


def test_zoom(camera):
    camera.enable_parallel_projection()
    orig_scale = camera.parallel_scale
    zoom = np.random.default_rng().random()
    camera.zoom(zoom)
    assert camera.parallel_scale == orig_scale / zoom


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
    near_point = np.random.default_rng().random()
    far_point = near_point + np.random.default_rng().random()
    points = (near_point, far_point)
    camera.clipping_range = points
    assert camera.GetClippingRange() == points
    assert camera.clipping_range == points

    far_point = near_point - np.random.default_rng().random()
    points = (near_point, far_point)
    with pytest.raises(ValueError):  # noqa: PT011
        camera.clipping_range = points


def test_reset_clipping_range(camera):
    with pytest.raises(AttributeError):
        camera.reset_clipping_range()

    # requires renderer for this method
    crng = (1, 2)
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.camera.clipping_range = crng
    assert pl.camera.clipping_range == crng
    pl.camera.reset_clipping_range()
    assert pl.camera.clipping_range != crng


def test_view_angle(camera):
    assert camera.GetViewAngle() == camera.view_angle
    view_angle = 60.0
    camera.view_angle = view_angle
    assert camera.GetViewAngle() == view_angle


def test_direction(camera):
    assert camera.GetDirectionOfProjection() == camera.direction


def test_view_frustum(camera):
    frustum = camera.view_frustum(1.0)
    assert frustum.n_points == 8
    assert frustum.n_cells == 6


def test_roll(camera):
    angle = 360.0 * (np.random.default_rng().random() - 0.5)
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


def test_eq():
    camera = pv.Camera()
    other = pv.Camera()
    for camera_now in camera, other:
        for name, value, _ in configuration:
            setattr(camera_now, name, value)

    assert camera == other

    # check that changing anything will break equality
    for name, value, _ in configuration:
        original_value = getattr(other, name)
        if isinstance(value, bool):
            changed_value = not value
        elif isinstance(value, (int, float)):
            changed_value = 0
        elif isinstance(value, tuple):
            changed_value = (0.5, 0.5, 0.5)
        else:
            changed_value = -value
        setattr(other, name, changed_value)
        assert camera != other
        setattr(other, name, original_value)

    # sanity check that we managed to restore the original state
    assert camera == other


def test_copy():
    camera = pv.Camera()
    for name, value, _ in configuration:
        setattr(camera, name, value)

    deep = camera.copy()
    assert deep == camera


def test_repr(camera):
    assert 'Camera' in repr(camera)
    assert 'Position' in repr(camera)
    assert 'Focal Point' in repr(camera)
    assert 'Parallel Projection' in repr(camera)
    assert 'Distance' in repr(camera)
    assert 'Thickness' in repr(camera)
    assert 'Parallel Scale' in repr(camera)
    assert 'Clipping Range' in repr(camera)
    assert 'View Angle' in repr(camera)
    assert 'Roll' in repr(camera)


def test_str(camera):
    assert 'Camera' in str(camera)
    assert 'Position' in str(camera)
    assert 'Focal Point' in str(camera)
    assert 'Parallel Projection' in str(camera)
    assert 'Distance' in str(camera)
    assert 'Thickness' in str(camera)
    assert 'Parallel Scale' in str(camera)
    assert 'Clipping Range' in str(camera)
    assert 'View Angle' in str(camera)
    assert 'Roll' in str(camera)
