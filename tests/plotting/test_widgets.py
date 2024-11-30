from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import VTKVersionError
from pyvista.plotting.affine_widget import DARK_YELLOW
from pyvista.plotting.affine_widget import get_angle
from pyvista.plotting.affine_widget import ray_plane_intersection
from tests.conftest import flaky_test

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def r_mat_to_euler_angles(R):
    """Extract Euler angles from a 3x3 rotation matrix using the ZYX sequence.

    Returns the angles in radians.
    """
    # Check for gimbal lock: singular cases
    if abs(R[2, 0]) == 1:
        # Gimbal lock exists
        yaw = 0  # Set yaw to 0 and compute the others
        if R[2, 0] == -1:  # Directly looking up
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:  # Directly looking down
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    else:
        # Not at the singularities
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    if np.isclose(roll, np.pi):
        roll = 0
    if np.isclose(pitch, np.pi):
        pitch = 0
    if np.isclose(yaw, np.pi):
        yaw = 0

    return roll, pitch, yaw


def test_widget_box(uniform):
    p = pv.Plotter()
    func = lambda box: box  # Does nothing
    p.add_mesh(uniform)
    p.add_box_widget(callback=func)
    p.close()

    p = pv.Plotter()
    func = lambda box, widget: box  # Does nothing
    p.add_mesh(uniform)
    p.add_box_widget(callback=func, pass_widget=True)
    p.close()

    # clip box with and without crinkle
    p = pv.Plotter()
    p.add_mesh_clip_box(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_box(uniform, crinkle=True)
    p.close()

    p = pv.Plotter()
    # merge_points=True is the default and is tested above
    p.add_mesh_clip_box(uniform, merge_points=False)
    p.close()


def test_widget_plane(uniform):
    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, implicit=False)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=False)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, assign_to_axis='z', implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, normal_rotation=False, implicit=False)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_plane(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_plane(uniform, crinkle=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice_orthogonal(uniform)
    p.close()


def test_widget_line(uniform):
    p = pv.Plotter()
    func = lambda line: line  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func)
    p.close()

    p = pv.Plotter()
    func = lambda line, widget: line  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func, pass_widget=True)
    p.close()

    p = pv.Plotter()
    func = lambda a, b: (a, b)  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func, use_vertices=True)
    p.close()


def test_widget_text_slider(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    with pytest.raises(TypeError, match='must be a list'):
        p.add_text_slider_widget(callback=func, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        p.add_text_slider_widget(callback=func, data=[])
    for style in pv.global_theme.slider_styles:
        p.add_text_slider_widget(callback=func, data=['foo', 'bar'], style=style)
    p.close()


def test_widget_slider(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    p.add_slider_widget(callback=func, rng=[0, 10], style='classic')
    p.close()

    p = pv.Plotter()
    for interaction_event in ['start', 'end', 'always']:
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event=interaction_event)
    with pytest.raises(TypeError, match='type for ``style``'):
        p.add_slider_widget(callback=func, rng=[0, 10], style=0)
    with pytest.raises(AttributeError):
        p.add_slider_widget(callback=func, rng=[0, 10], style='foo')
    with pytest.raises(TypeError, match='Expected type for `interaction_event`'):
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event=0)
    with pytest.raises(ValueError, match='Expected value for `interaction_event`'):
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event='foo')
    p.close()

    p = pv.Plotter()
    func = lambda value, widget: value  # Does nothing
    p.add_mesh(uniform)
    p.add_slider_widget(callback=func, rng=[0, 10], style='modern', pass_widget=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_threshold(uniform, invert=True)
    p.add_mesh(uniform.outline())
    p.close()

    p = pv.Plotter()
    p.add_mesh_threshold(uniform, invert=False)
    p.add_mesh(uniform.outline())
    p.close()

    p = pv.Plotter()
    p.add_mesh_isovalue(uniform)
    p.close()

    func = lambda value: value  # Does nothing
    p = pv.Plotter()
    title_height = np.random.default_rng().random()
    s = p.add_slider_widget(callback=func, rng=[0, 10], style='classic', title_height=title_height)
    assert s.GetRepresentation().GetTitleHeight() == title_height
    p.close()

    p = pv.Plotter()
    title_opacity = np.random.default_rng().random()
    s = p.add_slider_widget(
        callback=func,
        rng=[0, 10],
        style='classic',
        title_opacity=title_opacity,
    )
    assert s.GetRepresentation().GetTitleProperty().GetOpacity() == title_opacity
    p.close()

    p = pv.Plotter()
    title_color = 'red'
    s = p.add_slider_widget(callback=func, rng=[0, 10], style='classic', title_color=title_color)
    assert s.GetRepresentation().GetTitleProperty().GetColor() == pv.Color(title_color)
    p.close()

    p = pv.Plotter()
    fmt = '%0.9f'
    s = p.add_slider_widget(callback=func, rng=[0, 10], style='classic', fmt=fmt)
    assert s.GetRepresentation().GetLabelFormat() == fmt
    p.close()

    # custom width
    p = pv.Plotter()
    slider = p.add_slider_widget(
        callback=func,
        rng=[0, 10],
        fmt=fmt,
        tube_width=0.1,
        slider_width=0.2,
    )
    assert slider.GetRepresentation().GetSliderWidth() == 0.2
    assert slider.GetRepresentation().GetTubeWidth() == 0.1
    p.close()


def test_widget_spline(uniform):
    p = pv.Plotter()
    func = lambda spline: spline  # Does nothing
    p.add_mesh(uniform)
    p.add_spline_widget(callback=func)
    p.close()

    p = pv.Plotter()
    p.add_mesh(uniform)
    pts = np.array([[1, 5, 4], [2, 4, 9], [3, 6, 2]])
    with pytest.raises(ValueError, match='`initial_points` must be length `n_handles`'):
        p.add_spline_widget(callback=func, n_handles=4, initial_points=pts)
    p.add_spline_widget(callback=func, n_handles=3, initial_points=pts)
    p.close()

    p = pv.Plotter()
    func = lambda spline, widget: spline  # Does nothing
    p.add_mesh(uniform)
    p.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice_spline(uniform)
    p.close()


def test_measurement_widget(random_hills):
    class DistanceCallback:
        def __init__(self):
            self.called = False
            self.args = None
            self.count = 0

        def __call__(self, *args, **kwargs):
            self.called = True
            self.args = args
            self.kwargs = kwargs
            self.count += 1

    p = pv.Plotter(window_size=[1000, 1000])
    p.add_mesh(random_hills)
    distance_callback = DistanceCallback()
    p.add_measurement_widget(callback=distance_callback)
    p.view_xy()
    p.show(auto_close=False)
    width, height = p.window_size

    p.iren._mouse_left_button_click(300, 300)
    p.iren._mouse_left_button_click(700, 700)

    assert distance_callback.called
    assert pytest.approx(distance_callback.args[2], 1.0) == 17.4

    p.close()


def test_widget_sphere(uniform):
    p = pv.Plotter()
    func = lambda center: center  # Does nothing
    p.add_sphere_widget(callback=func, center=(0, 0, 0))
    p.close()

    # pass multiple centers
    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    p = pv.Plotter()
    func = lambda center, index: center  # Does nothing
    p.add_sphere_widget(callback=func, center=nodes)
    p.close()


def test_widget_checkbox_button(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    p.add_checkbox_button_widget(callback=func)
    p.close()


def test_widget_closed(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_checkbox_button_widget(callback=lambda value: value)


def test_widget_radio_button(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    b = p.add_radio_button_widget(callback=func, radio_button_group='group')
    assert p.radio_button_widget_dict['group'][0] == b
    p.close()
    assert 'group' not in p.radio_button_widget_dict


def test_widget_radio_button_click(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    size = 50
    position = (10.0, 10.0)
    b = p.add_radio_button_widget(
        callback=func, radio_button_group='group', value=False, size=size, position=position
    )
    p.show(auto_close=False)
    # Test switching logic
    b_center = (int(position[0] + size / 2), int(position[1] + size / 2))
    assert b.GetRepresentation().GetState() == 0
    p.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    p.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    p.close()


def test_widget_radio_button_with_title(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    p.add_radio_button_widget(callback=func, radio_button_group='group', title='my_button')
    assert len(p.radio_button_title_dict['group']) == 1
    p.close()
    assert 'group' not in p.radio_button_title_dict


def test_widget_radio_button_multiple_on(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    b1 = p.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    b2 = p.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    assert len(p.radio_button_widget_dict['group']) == 2
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    p.close()


def test_widget_radio_button_multiple_switch(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    size = 50
    b1_position = (10.0, 10.0)
    b2_position = (10.0, 70.0)
    b1 = p.add_radio_button_widget(
        callback=func, radio_button_group='group', value=True, size=size, position=b1_position
    )
    b2 = p.add_radio_button_widget(
        callback=func, radio_button_group='group', size=size, position=b2_position
    )
    p.show(auto_close=False)
    # Click b2 and switch active radio button
    b2_center = (int(b2_position[0] + size / 2), int(b2_position[1] + size / 2))
    p.iren._mouse_left_button_click(*b2_center)
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    p.close()


def test_widget_radio_button_plotter_closed(uniform):
    p = pv.Plotter()
    func = lambda: None  # Does nothing
    p.add_mesh(uniform)
    p.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        p.add_radio_button_widget(callback=func, radio_button_group='group')


@pytest.mark.needs_vtk_version(9, 1)
def test_add_camera_orientation_widget(uniform):
    p = pv.Plotter()
    p.add_camera_orientation_widget()
    assert p.camera_widgets
    p.close()
    assert not p.camera_widgets


def test_plot_algorithm_widgets():
    algo = vtk.vtkRTAnalyticSource()

    pl = pv.Plotter()
    pl.add_mesh_clip_box(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_isovalue(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(algo)
    pl.close()


def test_add_volume_clip_plane(uniform):
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='The `volume` parameter type must'):
        pl.add_volume_clip_plane(pv.Sphere())

    widget = pl.add_volume_clip_plane(uniform)
    assert isinstance(widget, vtk.vtkImplicitPlaneWidget)
    assert pl.volume.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()

    pl = pv.Plotter()
    vol = pl.add_volume(uniform)
    assert vol.mapper.GetClippingPlanes() is None
    pl.add_volume_clip_plane(vol)
    assert vol.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_plot_pointset_widgets(pointset):
    pointset = pointset.elevation()

    assert isinstance(pointset, pv.PointSet)

    pl = pv.Plotter()
    pl.add_mesh_clip_box(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(pointset)
    pl.close()


def test_ray_plane_intersection():
    # Test data
    start_point = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 5])
    normal = np.array([0, 0, 1])

    # Expected result
    expected_result = np.array([0, 0, 5])
    result = ray_plane_intersection(start_point, direction, plane_point, normal)
    np.testing.assert_array_almost_equal(result, expected_result)


def test_get_angle():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    # Expect 90 degrees between these two orthogonal vectors
    expected_angle = 90.0
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)

    # Test with parallel vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 3])
    expected_angle = 0.0  # angle between two identical vectors should be 0
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)

    # Test with opposite vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    expected_angle = 180.0  # angle between two opposite vectors should be 180
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)


@flaky_test
def test_affine_widget(sphere):
    interact_calls = []
    release_calls = []

    def interact_callback(transform):
        interact_calls.append(transform)

    def release_callback(transform):
        release_calls.append(transform)

    pl = pv.Plotter(window_size=(400, 400))
    actor = pl.add_mesh(sphere)

    if pv.vtk_version_info < (9, 2):
        with pytest.raises(VTKVersionError):
            pl.add_affine_transform_widget(actor)
        return

    with pytest.raises(TypeError, match='callable'):
        pl.add_affine_transform_widget(actor, interact_callback='foo')

    with pytest.raises(ValueError, match='(3, 3)'):
        pl.add_affine_transform_widget(actor, axes=np.eye(5))

    axes = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    with pytest.raises(ValueError, match='right hand'):
        pl.add_affine_transform_widget(actor, axes=axes)

    widget = pl.add_affine_transform_widget(
        actor,
        interact_callback=interact_callback,
        release_callback=release_callback,
    )
    pl.show(auto_close=False)

    assert not widget._selected_actor

    # move in the center and ensure that an actor is selected
    width, height = pl.window_size
    pl.iren._mouse_move(width // 2, height // 2)
    assert widget._selected_actor in widget._arrows + widget._circles
    assert widget._selected_actor.prop.color == pv.Color(DARK_YELLOW)

    # ensure that the actor gets deselected
    pl.iren._mouse_move(0, 0)
    assert not widget._selected_actor

    # test X axis translation
    pl.iren._mouse_left_button_press(width // 2 - 1, height // 2 - 1)
    assert widget._selected_actor is widget._arrows[0]
    assert widget._pressing_down
    pl.iren._mouse_move(width, height // 2)
    assert actor.user_matrix[0, 3] < 0
    pl.iren._mouse_left_button_release(width, height // 2)
    assert actor.user_matrix[0, 3] < 0

    # test callback called
    assert len(interact_calls) == 2
    assert interact_calls[0].shape == (4, 4)
    assert len(release_calls) == 1
    assert release_calls[0].shape == (4, 4)

    # test Y axis translation
    pl.iren._mouse_left_button_press(width // 2 + 1, height // 2 - 1)
    assert widget._selected_actor is widget._arrows[1]
    assert widget._pressing_down
    pl.iren._mouse_move(width, height // 2)
    assert actor.user_matrix[1, 3] < 0
    pl.iren._mouse_left_button_release()
    assert actor.user_matrix[1, 3] < 0

    # test Z axis translation
    pl.iren._mouse_left_button_press(width // 2, height // 2 + 5)
    assert widget._selected_actor is widget._arrows[2]
    assert widget._pressing_down
    pl.iren._mouse_move(width // 2, 0)
    assert actor.user_matrix[3, 3] > 0
    pl.iren._mouse_left_button_release()
    assert actor.user_matrix[3, 3] > 0

    # test X axis rotation
    pl.iren._mouse_left_button_press(width // 2 + 30, height // 2)
    assert widget._selected_actor is widget._circles[0]
    assert widget._pressing_down
    pl.iren._mouse_move(width // 2 + 30, height // 2 + 1)
    x_r, y_r, z_r = r_mat_to_euler_angles(actor.user_matrix)
    assert x_r > 0
    assert np.allclose([y_r, z_r], 0)
    pl.iren._mouse_left_button_release()
    assert not widget._pressing_down
    widget._reset()
    assert np.allclose(widget._cached_matrix, np.eye(4))

    # test Y axis rotation
    pl.iren._mouse_left_button_press(width // 2 - 30, height // 2)
    assert widget._selected_actor is widget._circles[1]
    assert widget._pressing_down
    pl.iren._mouse_move(width // 2 - 30, height // 2 - 1)
    x_r, y_r, z_r = r_mat_to_euler_angles(actor.user_matrix)
    assert y_r > 0
    assert np.allclose([x_r, z_r], 0)
    pl.iren._mouse_left_button_release()
    assert not widget._pressing_down
    widget._reset()

    # test Z axis rotation
    pl.iren._mouse_left_button_press(width // 2, height // 2 - 28)
    assert widget._selected_actor is widget._circles[2]
    assert widget._pressing_down
    pl.iren._mouse_move(width // 2 - 1, height // 2 + 30)
    x_r, y_r, z_r = r_mat_to_euler_angles(actor.user_matrix)
    assert z_r > 0
    assert np.allclose([x_r, y_r], 0)
    pl.iren._mouse_left_button_release()
    assert not widget._pressing_down
    widget._reset()

    # test change axes
    axes = np.array(
        [[0.70710678, 0.70710678, 0.0], [-0.70710678, 0.70710678, 0.0], [0.0, 0.0, 1.0]],
    )
    widget.axes = axes
    assert np.allclose(widget.axes, axes)

    # test X axis translation with new axes
    pl.iren._mouse_left_button_press(width // 2, height // 2 - 30)
    assert widget._selected_actor is widget._arrows[0]
    assert widget._pressing_down
    pl.iren._mouse_move(width // 2, height // 2 - 32)
    assert actor.user_matrix[0, 3] > 0
    pl.iren._mouse_left_button_release(width, height // 2 - 32)
    assert actor.user_matrix[0, 3] > 0

    # test origin
    origin = np.random.default_rng().random(3)
    widget.origin = origin
    assert np.allclose(widget.origin, origin)

    # test disable
    assert pl._picker_in_use
    widget.disable()
    assert not pl._picker_in_use

    widget.remove()
    assert not widget._circles
    assert not widget._arrows


def test_logo_widget(verify_image_cache):
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_logo_widget()
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_logo_widget(position=(0.01, 0.01), size=(0.8, 0.8))
    pl.show()

    pl = pv.Plotter()
    # has a 2 x 1 aspect ratio
    pl.add_logo_widget(examples.mapfile, position=(0.0, 0.0), size=(0.99, 0.495))
    pl.show()

    pl = pv.Plotter()
    pl.add_logo_widget(
        examples.download_vtk_logo().to_image(),
        position=(0.0, 0.0),
        size=(0.8, 0.8),
    )
    pl.show()

    pl = pv.Plotter()
    with pytest.raises(TypeError, match='must be a pyvista.ImageData or a file path'):
        pl.add_logo_widget(logo=0)


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_camera3d_widget(verify_image_cache):
    sphere = pv.Sphere()
    plotter = pv.Plotter(window_size=[600, 300], shape=(1, 2))
    plotter.add_mesh(sphere)
    plotter.subplot(0, 1)
    plotter.add_mesh(sphere)
    plotter.add_camera3d_widget()
    plotter.show(cpos=plotter.camera_position)


@pytest.mark.parametrize('outline_opacity', [True, False, np.random.default_rng(0).random()])
def test_outline_opacity(uniform, outline_opacity):
    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    plane_widget = p.add_plane_widget(callback=func, implicit=True, outline_opacity=outline_opacity)
    assert plane_widget.GetOutlineProperty().GetOpacity() == float(outline_opacity)
    p.close()


def test_clear_box_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_box_widget(None)
    pl.clear_box_widgets()
    pl.show(cpos='xy')


def test_clear_plane_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_plane_widget(None)
    pl.clear_plane_widgets()
    pl.show(cpos='xy')


def test_clear_line_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_line_widget(None)
    pl.clear_line_widgets()
    pl.show(cpos='xy')


def test_clear_slider_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_slider_widget(None, [0, 1])
    pl.clear_slider_widgets()
    pl.show(cpos='xy')


def test_clear_spline_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_spline_widget(None)
    pl.clear_spline_widgets()
    pl.show(cpos='xy')


def test_clear_measure_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_measurement_widget(None)
    pl.clear_measure_widgets()
    pl.show(cpos='xy')


def test_clear_sphere_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_sphere_widget(None)
    pl.clear_sphere_widgets()
    pl.show(cpos='xy')


@pytest.mark.needs_vtk_version(9, 1)
def test_clear_camera_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera_orientation_widget()
    pl.clear_camera_widgets()
    pl.show(cpos='xy')


def test_clear_button_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_checkbox_button_widget(None)
    pl.clear_button_widgets()
    pl.show(cpos='xy')


def test_clear_logo_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_logo_widget(None)
    pl.clear_logo_widgets()
    pl.show(cpos='xy')


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_clear_camera3d_widget(verify_image_cache):
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera3d_widget()
    pl.clear_camera3d_widgets()
    pl.show(cpos='xy')
