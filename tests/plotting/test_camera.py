from __future__ import annotations

from io import StringIO
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import transformations

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


def test_eq_not_a_camera():
    assert pv.Camera() != 5


def test_tight_without_renderer():
    match = 'Camera must be associated with a renderer to fit it to the actors.'
    with pytest.raises(AttributeError, match=re.escape(match)):
        pv.Camera().tight()


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


@pytest.fixture
def extrinsics():
    """Return an extrinsic matrix with no axis left unrotated."""
    rotation_vector = np.array([0.15, -0.35, 0.05])
    matrix = np.eye(4)
    matrix[:3, :3] = transformations.axis_angle_rotation(
        rotation_vector, np.linalg.norm(rotation_vector), deg=False
    )[:3, :3]
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


@pytest.fixture
def calibrated():
    """Return a plotter whose window is the size the intrinsics are calibrated for."""
    pl = pv.Plotter(window_size=IMAGE_SIZE)
    yield pl
    pl.close()


def test_intrinsic_matrix_round_trip(calibrated):
    """An intrinsic matrix is recovered exactly after being set."""
    calibrated.camera.intrinsic_matrix = INTRINSICS
    assert calibrated.camera.intrinsic_matrix == pytest.approx(INTRINSICS)


def test_intrinsic_matrix_marks_the_camera_set(calibrated):
    """Setting intrinsics keeps the first render from resetting the view angle."""
    camera = calibrated.renderer.camera
    assert not camera.is_set
    camera.intrinsic_matrix = INTRINSICS
    assert camera.is_set


def test_intrinsic_matrix_is_discarded_by_a_camera_reset(calibrated):
    """A camera reset restores the default field of view but keeps the principal point."""
    calibrated.add_mesh(pv.Sphere())
    calibrated.camera.intrinsic_matrix = INTRINSICS
    calibrated.reset_camera()
    reset = calibrated.camera.intrinsic_matrix
    assert reset[0, 0] != pytest.approx(INTRINSICS[0, 0])
    assert reset[1, 1] != pytest.approx(INTRINSICS[1, 1])
    assert reset[:2, 2] == pytest.approx(INTRINSICS[:2, 2])


def test_linked_views_keep_the_camera_on_its_own_viewport():
    """Linking views leaves a shared camera calibrated for the renderer it came from."""
    pl = pv.Plotter(shape='1|2')
    pl.subplot(0)
    pl.camera.intrinsic_matrix = INTRINSICS
    before = pl.camera.intrinsic_matrix
    pl.link_views()
    assert pl.camera.intrinsic_matrix == pytest.approx(before)
    pl.close()


def test_intrinsic_matrix_follows_the_window(calibrated):
    """The reported focal length scales with the window the camera renders into."""
    calibrated.camera.intrinsic_matrix = INTRINSICS
    width, height = IMAGE_SIZE
    calibrated.window_size = (2 * width, 3 * height)
    calibrated.render()
    assert calibrated.camera.intrinsic_matrix == pytest.approx(
        np.diag([2.0, 3.0, 1.0]) @ INTRINSICS
    )


def test_intrinsic_matrix_uses_the_subplot_viewport():
    """A subplot is calibrated for its own viewport, not the whole window."""
    pl = pv.Plotter(shape=(1, 2), window_size=IMAGE_SIZE)
    pl.subplot(0, 1)
    intrinsics = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    pl.camera.intrinsic_matrix = intrinsics
    assert tuple(pl.renderer.GetSize()) == (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1])
    assert pl.camera.intrinsic_matrix == pytest.approx(intrinsics)
    pl.close()


def test_intrinsic_matrix_reaches_an_assigned_camera(calibrated):
    """A camera handed to a plotter is calibrated for that plotter."""
    calibrated.camera = pv.Camera()
    calibrated.camera.intrinsic_matrix = INTRINSICS
    assert calibrated.camera.intrinsic_matrix == pytest.approx(INTRINSICS)


def test_intrinsic_matrix_raises_without_a_plotter(camera):
    """A camera of its own has no image to be calibrated for."""
    with pytest.raises(RuntimeError, match='requires a plotter'):
        camera.intrinsic_matrix  # noqa: B018
    with pytest.raises(RuntimeError, match='requires a plotter'):
        camera.intrinsic_matrix = INTRINSICS
    with pytest.raises(RuntimeError, match='requires a plotter'):
        getattr(camera, 'intrinsic_matrix', None)


def test_intrinsic_matrix_raises_for_a_closed_plotter(calibrated):
    """A closed plotter leaves the camera with no viewport to be calibrated for."""
    camera = calibrated.camera
    calibrated.close()
    with pytest.raises(RuntimeError, match='non-empty viewport, got 0x0'):
        camera.intrinsic_matrix  # noqa: B018


def test_intrinsic_matrix_disables_parallel_projection(calibrated):
    """Setting intrinsics gives the camera a perspective projection."""
    calibrated.camera.enable_parallel_projection()
    calibrated.camera.intrinsic_matrix = INTRINSICS
    assert not calibrated.camera.parallel_projection


def test_intrinsic_matrix_raises_for_parallel_projection(calibrated):
    """A parallel projection has no intrinsic matrix."""
    calibrated.camera.enable_parallel_projection()
    with pytest.raises(ValueError, match='perspective projection'):
        calibrated.camera.intrinsic_matrix  # noqa: B018


def test_intrinsic_matrix_raises_for_skew(calibrated):
    """Axis skew cannot be represented and is rejected."""
    skewed = INTRINSICS.copy()
    skewed[0, 1] = 1e-3
    with pytest.raises(ValueError, match='axis skew'):
        calibrated.camera.intrinsic_matrix = skewed


@pytest.mark.parametrize('entry', [(0, 0), (1, 1)], ids=['fx', 'fy'])
@pytest.mark.parametrize('focal_length', [0.0, -800.0])
def test_intrinsic_matrix_raises_for_focal_length(calibrated, entry, focal_length):
    """A focal length that is not positive is rejected."""
    invalid = INTRINSICS.copy()
    invalid[entry] = focal_length
    with pytest.raises(ValueError, match='focal lengths must be positive'):
        calibrated.camera.intrinsic_matrix = invalid


def test_intrinsic_matrix_raises_for_shape(calibrated):
    """A matrix that is not 3x3 is rejected."""
    with pytest.raises(ValueError, match='intrinsic matrix'):
        calibrated.camera.intrinsic_matrix = np.eye(4)


@pytest.mark.parametrize(
    'invalid',
    [INTRINSICS.T, np.vstack([INTRINSICS[:2], [7.0, 9.0, 4.0]]), np.tril(INTRINSICS.T)],
    ids=['transposed', 'last_row', 'lower_triangular'],
)
def test_intrinsic_matrix_raises_for_a_matrix_that_is_not_upper_triangular(calibrated, invalid):
    """A matrix that is not a pinhole intrinsic matrix is rejected."""
    with pytest.raises(ValueError, match='must be upper triangular'):
        calibrated.camera.intrinsic_matrix = invalid


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


def test_extrinsic_matrix_raises_for_a_scaled_rotation(camera, extrinsics):
    """An extrinsic matrix whose upper block is not orthogonal is rejected."""
    extrinsics[:3, :3] *= 2.0
    with pytest.raises(ValueError, match='must be orthogonal'):
        camera.extrinsic_matrix = extrinsics


def test_extrinsic_matrix_raises_for_a_reflection(camera, extrinsics):
    """An extrinsic matrix whose upper block is left-handed is rejected."""
    extrinsics[:3, :3] = np.diag([1.0, 1.0, -1.0]) @ extrinsics[:3, :3]
    with pytest.raises(ValueError, match='incorrect handedness'):
        camera.extrinsic_matrix = extrinsics


def test_calibrated_camera_matches_opencv_projection(calibrated, extrinsics):
    """The camera's own matrices project points where OpenCV projects them."""
    camera = calibrated.camera
    camera.intrinsic_matrix = INTRINSICS
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


def test_calibrated_camera_displays_where_opencv_projects(calibrated, extrinsics):
    """The renderer maps world points to the pixels OpenCV projects them to."""
    pl = calibrated
    pl.camera.intrinsic_matrix = INTRINSICS
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

    assert np.array(displayed) == pytest.approx(
        opencv_project(WORLD_POINTS, INTRINSICS, extrinsics)
    )


def test_copy_carries_the_calibrated_projection(calibrated):
    """A copied camera keeps the window center and explicit aspect ratio."""
    camera = calibrated.camera
    camera.intrinsic_matrix = INTRINSICS
    copied = camera.copy()
    assert copied.window_center == camera.window_center
    assert copied.explicit_aspect_ratio == camera.explicit_aspect_ratio
    assert copied == camera
