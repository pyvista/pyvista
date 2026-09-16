from __future__ import annotations

from io import StringIO
import re

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
        StringIO(tmp),
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


@pytest.mark.parametrize('zero', [(0, 0, 0), (0.0, 0.0, 0.0), np.zeros(3)])
def test_up_raises_zero_vector(camera, zero):
    original = camera.up
    with pytest.raises(ValueError, match=re.escape('Camera up vector cannot be zero.')):
        camera.up = zero
    assert camera.up == original


def test_camera_position_raises_zero_viewup():
    # VTK used to silently substitute (0, 1, 0) here. See #7826.
    pl = pv.Plotter()
    with pytest.raises(ValueError, match=re.escape('Camera up vector cannot be zero.')):
        pl.camera_position = [(15, 3, 15), (0, 0, 0), (0, 0, 0)]


def test_set_viewup_raises_zero_vector():
    pl = pv.Plotter()
    with pytest.raises(ValueError, match=re.escape('Camera up vector cannot be zero.')):
        pl.set_viewup((0, 0, 0))


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


CAMERA_REPR_FIELDS = [
    'Camera',
    'Position',
    'Focal Point',
    'Parallel Projection',
    'Distance',
    'Thickness',
    'Parallel Scale',
    'Clipping Range',
    'View Angle',
    'Roll',
]


@pytest.mark.parametrize('render', [repr, str], ids=['repr', 'str'])
def test_repr_and_str(camera, render):
    """Both text forms report every camera field."""
    text = render(camera)
    missing = [field for field in CAMERA_REPR_FIELDS if field not in text]
    assert not missing, f'Missing from {render.__name__}: {missing}'


IMAGE_SIZE = (640, 480)
INTRINSICS = np.array([[800.0, 0.0, 310.0], [0.0, 760.0, 250.0], [0.0, 0.0, 1.0]])


def rodrigues(rotation_vector):
    """Return the rotation matrix of an axis-angle vector."""
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


@pytest.fixture
def extrinsics():
    """Return an extrinsic matrix with no axis left unrotated."""
    matrix = np.eye(4)
    matrix[:3, :3] = rodrigues(np.array([0.15, -0.35, 0.05]))
    matrix[:3, 3] = (0.2, -0.1, 6.0)
    return matrix


def opencv_project(points, intrinsic_matrix, extrinsic_matrix):
    """Project world points to pixels with the OpenCV pinhole model."""
    camera_points = points @ extrinsic_matrix[:3, :3].T + extrinsic_matrix[:3, 3]
    pixels = camera_points @ intrinsic_matrix.T
    return pixels[:, :2] / pixels[:, 2:3]


WORLD_POINTS = np.array([[0.0, 0.0, 0.0], [0.4, 0.3, -0.2], [-0.5, 0.25, 0.35], [1.0, -0.8, 0.6]])


def test_window_center(camera):
    """The window center round-trips through the property."""
    assert camera.window_center == (0.0, 0.0)
    camera.window_center = (0.25, -0.1)
    assert camera.window_center == (0.25, -0.1)
    assert camera.GetWindowCenter() == (0.25, -0.1)


def test_window_center_raises(camera):
    """A window center of the wrong length is rejected."""
    with pytest.raises(ValueError, match='window center'):
        camera.window_center = (0.25, -0.1, 0.0)


def test_explicit_aspect_ratio(camera):
    """The explicit aspect ratio is ``None`` until set, and again once cleared."""
    assert camera.explicit_aspect_ratio is None
    camera.explicit_aspect_ratio = 1.25
    assert camera.explicit_aspect_ratio == 1.25
    assert camera.GetUseExplicitAspectRatio()
    camera.explicit_aspect_ratio = None
    assert camera.explicit_aspect_ratio is None
    assert not camera.GetUseExplicitAspectRatio()


@pytest.mark.parametrize('ratio', [0.0, -1.0])
def test_explicit_aspect_ratio_raises(camera, ratio):
    """A non-positive explicit aspect ratio is rejected."""
    with pytest.raises(ValueError, match='explicit aspect ratio'):
        camera.explicit_aspect_ratio = ratio


def test_intrinsic_matrix_round_trip(camera):
    """An intrinsic matrix is recovered exactly after being set."""
    camera.set_intrinsic_matrix(INTRINSICS, IMAGE_SIZE)
    assert camera.get_intrinsic_matrix(IMAGE_SIZE) == pytest.approx(INTRINSICS)
    assert camera.is_set


def test_from_intrinsics():
    """The constructor matches setting the matrix on a default camera."""
    camera = pv.Camera.from_intrinsics(INTRINSICS, IMAGE_SIZE)
    other = pv.Camera()
    other.set_intrinsic_matrix(INTRINSICS, IMAGE_SIZE)
    assert camera == other


def test_set_intrinsic_matrix_disables_parallel_projection(camera):
    """Setting intrinsics gives the camera a perspective projection."""
    camera.enable_parallel_projection()
    camera.set_intrinsic_matrix(INTRINSICS, IMAGE_SIZE)
    assert not camera.parallel_projection


def test_get_intrinsic_matrix_raises_for_parallel_projection(camera):
    """A parallel projection has no intrinsic matrix."""
    camera.enable_parallel_projection()
    with pytest.raises(ValueError, match='perspective projection'):
        camera.get_intrinsic_matrix(IMAGE_SIZE)


def test_set_intrinsic_matrix_raises_for_skew(camera):
    """Axis skew cannot be represented and is rejected."""
    skewed = INTRINSICS.copy()
    skewed[0, 1] = 1e-3
    with pytest.raises(ValueError, match='axis skew'):
        camera.set_intrinsic_matrix(skewed, IMAGE_SIZE)


@pytest.mark.parametrize('focal_length', [0.0, -800.0])
def test_set_intrinsic_matrix_raises_for_focal_length(camera, focal_length):
    """A focal length that is not positive is rejected."""
    invalid = INTRINSICS.copy()
    invalid[0, 0] = focal_length
    with pytest.raises(ValueError, match='focal lengths must be positive'):
        camera.set_intrinsic_matrix(invalid, IMAGE_SIZE)


@pytest.mark.parametrize('image_size', [(640, 0), (640.5, 480), (640, 480, 3)])
def test_intrinsic_matrix_raises_for_image_size(camera, image_size):
    """An image size that is not two positive integers is rejected."""
    with pytest.raises(ValueError, match='image size'):
        camera.set_intrinsic_matrix(INTRINSICS, image_size)


def test_extrinsic_matrix(camera):
    """A camera looking down ``-z`` has its own axes flipped in ``y`` and ``z``."""
    camera.position = (0.0, 0.0, 4.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 1.0, 0.0)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert camera.extrinsic_matrix == pytest.approx(expected)


def test_extrinsic_matrix_round_trip(camera, extrinsics):
    """An extrinsic matrix is recovered exactly after being set."""
    camera.extrinsic_matrix = extrinsics
    assert camera.extrinsic_matrix == pytest.approx(extrinsics)


def test_extrinsic_matrix_keeps_distance(camera, extrinsics):
    """Setting an extrinsic matrix leaves the distance to the focal point alone."""
    camera.position = (0.0, 0.0, 7.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.extrinsic_matrix = extrinsics
    assert camera.distance == pytest.approx(7.0)


def test_extrinsic_matrix_raises_for_non_rotation(camera, extrinsics):
    """An extrinsic matrix whose upper block is not a rotation is rejected."""
    extrinsics[:3, :3] *= 2.0
    with pytest.raises(ValueError, match='extrinsic matrix rotation'):
        camera.extrinsic_matrix = extrinsics


def test_calibrated_camera_matches_opencv_projection(extrinsics):
    """The camera's own matrices project points where OpenCV projects them."""
    camera = pv.Camera.from_intrinsics(INTRINSICS, IMAGE_SIZE)
    camera.extrinsic_matrix = extrinsics
    camera.clipping_range = (0.1, 100.0)

    width, height = IMAGE_SIZE
    composite = pv.array_from_vtkmatrix(
        camera.GetCompositeProjectionTransformMatrix(width / height, *camera.clipping_range)
    )
    homogeneous = np.column_stack([WORLD_POINTS, np.ones(len(WORLD_POINTS))])
    clip = homogeneous @ composite.T
    normalized = clip[:, :2] / clip[:, 3:4]
    pixels = np.column_stack(
        [(normalized[:, 0] + 1.0) * width / 2, (1.0 - normalized[:, 1]) * height / 2]
    )
    assert pixels == pytest.approx(opencv_project(WORLD_POINTS, INTRINSICS, extrinsics))


def test_calibrated_camera_renders_where_opencv_projects(extrinsics):
    """A render window of the calibrated size maps world points to the same pixels."""
    pl = pv.Plotter(window_size=IMAGE_SIZE)
    pl.add_mesh(pv.Sphere(radius=0.5))
    pl.camera = pv.Camera.from_intrinsics(INTRINSICS, IMAGE_SIZE)
    pl.camera.extrinsic_matrix = extrinsics
    pl.camera.clipping_range = (0.1, 100.0)
    pl.render()
    assert tuple(pl.window_size) == IMAGE_SIZE

    displayed = []
    for point in WORLD_POINTS:
        pl.renderer.SetWorldPoint(*point, 1.0)
        pl.renderer.WorldToDisplay()
        display_x, display_y, _ = pl.renderer.GetDisplayPoint()
        displayed.append((display_x, IMAGE_SIZE[1] - display_y))
    pl.close()

    assert np.array(displayed) == pytest.approx(
        opencv_project(WORLD_POINTS, INTRINSICS, extrinsics)
    )


def test_copy_carries_the_calibration():
    """A copied camera keeps the window center and explicit aspect ratio."""
    camera = pv.Camera.from_intrinsics(INTRINSICS, IMAGE_SIZE)
    copied = camera.copy()
    assert copied.window_center == camera.window_center
    assert copied.explicit_aspect_ratio == camera.explicit_aspect_ratio
    assert copied == camera
