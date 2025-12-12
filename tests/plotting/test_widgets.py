from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING
from typing import Literal
from unittest.mock import ANY

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.plotting import _vtk
from pyvista.plotting import widgets
from pyvista.plotting.affine_widget import DARK_YELLOW
from pyvista.plotting.affine_widget import get_angle
from pyvista.plotting.affine_widget import ray_plane_intersection
from tests.conftest import flaky_test

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def r_mat_to_euler_angles(r):
    """Extract Euler angles from a 3x3 rotation matrix using the ZYX sequence.

    Returns the angles in radians.
    """
    # Check for gimbal lock: singular cases
    if abs(r[2, 0]) == 1:
        # Gimbal lock exists
        yaw = 0  # Set yaw to 0 and compute the others
        if r[2, 0] == -1:  # Directly looking up
            pitch = np.pi / 2
            roll = yaw + np.arctan2(r[0, 1], r[0, 2])
        else:  # Directly looking down
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-r[0, 1], -r[0, 2])
    else:
        # Not at the singularities
        pitch = -np.arcsin(r[2, 0])
        roll = np.arctan2(r[2, 1] / np.cos(pitch), r[2, 2] / np.cos(pitch))
        yaw = np.arctan2(r[1, 0] / np.cos(pitch), r[0, 0] / np.cos(pitch))

    if np.isclose(roll, np.pi):
        roll = 0
    if np.isclose(pitch, np.pi):
        pitch = 0
    if np.isclose(yaw, np.pi):
        yaw = 0

    return roll, pitch, yaw


def test_add_plane_widget_raises():
    pl = pv.Plotter()
    with pytest.raises(RuntimeError, match='assign_to_axis not understood'):
        pl.add_plane_widget(lambda *x: True, assign_to_axis='foo')  # noqa: ARG005


def test_add_slider_widget_raises():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RuntimeError, match=r'Cannot add a widget to a closed plotter.'):
        pl.add_slider_widget(lambda *x: True, rng=[0, 1])  # noqa: ARG005


def test_add_mesh_threshold_raises():
    pl = pv.Plotter()
    with pytest.raises(
        TypeError, match=r'MultiBlock datasets are not supported for threshold widget.'
    ):
        pl.add_mesh_threshold(mesh=pv.MultiBlock())

    pl = pv.Plotter()
    with pytest.raises(ValueError, match=r'No arrays present to threshold.'):
        pl.add_mesh_threshold(mesh=pv.PolyData())


def test_add_mesh_isovalue_raises():
    pl = pv.Plotter()
    with pytest.raises(TypeError, match=r'MultiBlock datasets are not supported for this widget.'):
        pl.add_mesh_isovalue(mesh=pv.MultiBlock())

    pl = pv.Plotter()
    with pytest.raises(
        ValueError, match=r'Input dataset for the contour filter must have data arrays.'
    ):
        pl.add_mesh_isovalue(mesh=pv.PolyData())

    pl = pv.Plotter()
    sp = pv.Sphere()
    sp.cell_data['foo'] = 1
    match = re.escape('Contour filter only works on Point data. Array (foo) is in the Cell data.')
    with pytest.raises(TypeError, match=match):
        pl.add_mesh_isovalue(mesh=sp, scalars='foo')


def test_add_mesh_isovalue_pointset_raises():
    pl = pv.Plotter()
    with pytest.raises(
        TypeError, match=r'PointSets are 0-dimensional and thus cannot produce contours.'
    ):
        pl.add_mesh_isovalue(mesh=pv.PointSet())


def test_add_measurement_widget_raises():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RuntimeError, match=r'Cannot add a widget to a closed plotter.'):
        pl.add_measurement_widget()


def test_widget_box(uniform):
    pl = pv.Plotter()
    func = lambda box: box  # Does nothing
    pl.add_mesh(uniform)
    pl.add_box_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    func = lambda box, widget: box  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_box_widget(callback=func, pass_widget=True)
    pl.close()

    # clip box with and without crinkle
    pl = pv.Plotter()
    pl.add_mesh_clip_box(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_box(uniform, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    # merge_points=True is the default and is tested above
    pl.add_mesh_clip_box(uniform, merge_points=False)
    pl.close()


def test_widget_plane(uniform):
    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, implicit=False)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, pass_widget=True, implicit=False)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, assign_to_axis='z', implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, normal_rotation=False, implicit=False)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(uniform, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_orthogonal(uniform)
    pl.close()


def test_widget_line(uniform):
    pl = pv.Plotter()
    func = lambda line: line  # Does nothing
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    func = lambda line, widget: line  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func, pass_widget=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda a, b: (a, b)  # Does nothing
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func, use_vertices=True)
    pl.close()


def test_widget_text_slider(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    with pytest.raises(TypeError, match='must be a list'):
        pl.add_text_slider_widget(callback=func, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        pl.add_text_slider_widget(callback=func, data=[])
    for style in pv.global_theme.slider_styles:
        pl.add_text_slider_widget(callback=func, data=['foo', 'bar'], style=style)
    pl.close()


def test_widget_slider(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    pl.add_slider_widget(callback=func, rng=[0, 10], style='classic')
    pl.close()

    pl = pv.Plotter()
    for interaction_event in ['start', 'end', 'always']:
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event=interaction_event)
    with pytest.raises(TypeError, match='type for ``style``'):
        pl.add_slider_widget(callback=func, rng=[0, 10], style=0)
    with pytest.raises(AttributeError):
        pl.add_slider_widget(callback=func, rng=[0, 10], style='foo')
    with pytest.raises(TypeError, match='Expected type for `interaction_event`'):
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event=0)
    with pytest.raises(ValueError, match='Expected value for `interaction_event`'):
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event='foo')
    pl.close()

    pl = pv.Plotter()
    func = lambda value, widget: value  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_slider_widget(callback=func, rng=[0, 10], style='modern', pass_widget=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(uniform, invert=True)
    pl.add_mesh(uniform.outline())
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(uniform, invert=False)
    pl.add_mesh(uniform.outline())
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_isovalue(uniform)
    pl.close()

    func = lambda value: value  # Does nothing
    pl = pv.Plotter()
    title_height = np.random.default_rng().random()
    s = pl.add_slider_widget(
        callback=func, rng=[0, 10], style='classic', title_height=title_height
    )
    assert s.GetRepresentation().GetTitleHeight() == title_height
    pl.close()

    pl = pv.Plotter()
    title_opacity = np.random.default_rng().random()
    s = pl.add_slider_widget(
        callback=func,
        rng=[0, 10],
        style='classic',
        title_opacity=title_opacity,
    )
    assert s.GetRepresentation().GetTitleProperty().GetOpacity() == title_opacity
    pl.close()

    pl = pv.Plotter()
    title_color = 'red'
    s = pl.add_slider_widget(callback=func, rng=[0, 10], style='classic', title_color=title_color)
    assert s.GetRepresentation().GetTitleProperty().GetColor() == pv.Color(title_color)
    pl.close()

    pl = pv.Plotter()
    fmt = '%0.9f'
    s = pl.add_slider_widget(callback=func, rng=[0, 10], style='classic', fmt=fmt)
    assert s.GetRepresentation().GetLabelFormat() == fmt
    pl.close()

    # custom width
    pl = pv.Plotter()
    slider = pl.add_slider_widget(
        callback=func,
        rng=[0, 10],
        fmt=fmt,
        tube_width=0.1,
        slider_width=0.2,
    )
    assert slider.GetRepresentation().GetSliderWidth() == 0.2
    assert slider.GetRepresentation().GetTubeWidth() == 0.1
    pl.close()


def test_widget_spline(uniform):
    pl = pv.Plotter()
    func = lambda spline: spline  # Does nothing
    pl.add_mesh(uniform)
    pl.add_spline_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pts = np.array([[1, 5, 4], [2, 4, 9], [3, 6, 2]])
    with pytest.raises(ValueError, match='`initial_points` must be length `n_handles`'):
        pl.add_spline_widget(callback=func, n_handles=4, initial_points=pts)
    pl.add_spline_widget(callback=func, n_handles=3, initial_points=pts)
    pl.close()

    pl = pv.Plotter()
    func = lambda spline, widget: spline  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(uniform)
    pl.close()


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

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_mesh(random_hills)
    distance_callback = DistanceCallback()
    pl.add_measurement_widget(callback=distance_callback)
    pl.view_xy()
    pl.show(auto_close=False)
    _width, _height = pl.window_size

    pl.iren._mouse_left_button_click(300, 300)
    pl.iren._mouse_left_button_click(700, 700)

    assert distance_callback.called
    assert pytest.approx(distance_callback.args[2], 1.0) == 17.4

    pl.close()


def test_widget_sphere():
    pl = pv.Plotter()
    func = lambda center: center  # Does nothing
    pl.add_sphere_widget(callback=func, center=(0, 0, 0))
    pl.close()

    # pass multiple centers
    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    pl = pv.Plotter()
    func = lambda center, index: center  # Does nothing  # noqa: ARG005
    pl.add_sphere_widget(callback=func, center=nodes)
    pl.close()


def test_widget_checkbox_button(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    pl.add_checkbox_button_widget(callback=func)
    pl.close()


def test_widget_closed(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_checkbox_button_widget(callback=lambda value: value)


def test_widget_radio_button(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    b = pl.add_radio_button_widget(callback=func, radio_button_group='group')
    assert pl.radio_button_widget_dict['group'][0] == b
    pl.close()
    assert 'group' not in pl.radio_button_widget_dict


def test_widget_radio_button_click(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    size = 50
    position = (10.0, 10.0)
    b = pl.add_radio_button_widget(
        callback=func,
        radio_button_group='group',
        value=False,
        size=size,
        position=position,
    )
    pl.show(auto_close=False)
    # Test switching logic
    b_center = (int(position[0] + size / 2), int(position[1] + size / 2))
    assert b.GetRepresentation().GetState() == 0
    pl.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    pl.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_with_title(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    pl.add_radio_button_widget(callback=func, radio_button_group='group', title='my_button')
    assert len(pl.radio_button_title_dict['group']) == 1
    pl.close()
    assert 'group' not in pl.radio_button_title_dict


def test_widget_radio_button_multiple_on(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    b1 = pl.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    b2 = pl.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    assert len(pl.radio_button_widget_dict['group']) == 2
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_multiple_switch(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    size = 50
    b1_position = (10.0, 10.0)
    b2_position = (10.0, 70.0)
    b1 = pl.add_radio_button_widget(
        callback=func,
        radio_button_group='group',
        value=True,
        size=size,
        position=b1_position,
    )
    b2 = pl.add_radio_button_widget(
        callback=func, radio_button_group='group', size=size, position=b2_position
    )
    pl.show(auto_close=False)
    # Click b2 and switch active radio button
    b2_center = (int(b2_position[0] + size / 2), int(b2_position[1] + size / 2))
    pl.iren._mouse_left_button_click(*b2_center)
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_plotter_closed(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_radio_button_widget(callback=func, radio_button_group='group')


def test_add_camera_orientation_widget():
    pl = pv.Plotter()
    pl.add_camera_orientation_widget()
    assert pl.camera_widgets
    pl.close()
    assert not pl.camera_widgets


def test_plot_algorithm_widgets():
    algo = _vtk.vtkRTAnalyticSource()

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
    assert isinstance(widget, _vtk.vtkImplicitPlaneWidget)
    assert pl.volume.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()

    pl = pv.Plotter()
    vol = pl.add_volume(uniform)
    assert vol.mapper.GetClippingPlanes() is None
    pl.add_volume_clip_plane(vol)
    assert vol.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()


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
    result = ray_plane_intersection(
        start_point=start_point, direction=direction, plane_point=plane_point, normal=normal
    )
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

    with pytest.raises(TypeError, match='callable'):
        pl.add_affine_transform_widget(actor, interact_callback='foo')

    with pytest.raises(ValueError, match=r'(3, 3)'):
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

    def test_translation(
        press_pos: tuple[float, float],
        move_pos: tuple[float, float],
        idx: int,
        direction: Literal['neg', 'pos'],
    ):
        pl.iren._mouse_left_button_press(*press_pos)
        assert widget._selected_actor is widget._arrows[idx]
        assert widget._pressing_down

        pl.iren._mouse_move(*move_pos)
        trans_val = actor.user_matrix[idx, 3]
        assert (trans_val < 0) if direction == 'neg' else (trans_val > 0)

        pl.iren._mouse_left_button_release(*move_pos)
        trans_val = actor.user_matrix[idx, 3]
        assert (trans_val < 0) if direction == 'neg' else (trans_val > 0)
        assert not widget._pressing_down
        widget._reset()
        assert np.allclose(widget._cached_matrix, np.eye(4))

    # test X axis translation
    test_translation(
        press_pos=(width // 2 - 4, height // 2 - 4),
        move_pos=(width, height // 2),
        idx=0,
        direction='neg',
    )

    # test callback called
    assert len(interact_calls) == 2
    assert interact_calls[0].shape == (4, 4)
    assert len(release_calls) == 1
    assert release_calls[0].shape == (4, 4)

    # test Y axis translation
    test_translation(
        press_pos=(width // 2 + 2, height // 2 - 3),
        move_pos=(width, height // 2),
        idx=1,
        direction='pos',
    )

    # test Z axis translation
    test_translation(
        press_pos=(width // 2, height // 2 + 5),
        move_pos=(width // 2, 0),
        idx=2,
        direction='neg',
    )

    def test_rotation(
        press_pos: tuple[float, float],
        move_pos: tuple[float, float],
        idx: int,
        rotation: Literal['ccw', 'cw'],
    ):
        pl.iren._mouse_left_button_press(*press_pos)
        assert widget._selected_actor is widget._circles[idx]
        assert widget._pressing_down
        pl.iren._mouse_move(*move_pos)
        euler_angles = r_mat_to_euler_angles(actor.user_matrix)
        assert euler_angles[idx] > 0 if rotation == 'ccw' else euler_angles[idx] < 0
        assert np.count_nonzero(np.isclose(euler_angles, 0)) == len(euler_angles) - 1
        pl.iren._mouse_left_button_release()
        assert not widget._pressing_down
        widget._reset()
        assert np.allclose(widget._cached_matrix, np.eye(4))

    # test X axis rotation, counterclockwise
    test_rotation(
        press_pos=(width // 2 + 30, height // 2),
        move_pos=(width // 2 + 21, height // 2 + 17),
        idx=0,
        rotation='ccw',
    )

    # test Y axis rotation, counterclockwise
    test_rotation(
        press_pos=(width // 2 - 20, height // 2 + 20),
        move_pos=(width // 2 - 30, height // 2 + 4),
        idx=1,
        rotation='ccw',
    )

    # test Z axis rotation, clockwise
    test_rotation(
        press_pos=(width // 2, height // 2 - 29),
        move_pos=(width // 2 - 12, height // 2 - 23),
        idx=2,
        rotation='cw',
    )

    # test change axes
    axes = np.array(
        [
            [0.70710678, 0.70710678, 0.0],
            [-0.70710678, 0.70710678, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    widget.axes = axes
    assert np.allclose(widget.axes, axes)

    # test X axis translation with new axes
    test_translation(
        press_pos=(width // 2, height // 2 - 38),
        move_pos=(width // 2, height // 2 - 50),
        idx=0,
        direction='pos',
    )

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

    interact_calls = []
    release_calls = []


@pytest.mark.usefixtures('verify_image_cache')
def test_logo_widget():
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
    with pytest.raises(TypeError, match=r'must be a pyvista.ImageData or a file path'):
        pl.add_logo_widget(logo=0)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason='MacOS CI produces a slightly different camera position. Needs investigation.',
)
@pytest.mark.needs_vtk_version(9, 3, 0)
@pytest.mark.usefixtures('verify_image_cache')
def test_camera3d_widget():
    sphere = pv.Sphere()
    pl = pv.Plotter(window_size=[600, 300], shape=(1, 2))
    pl.add_mesh(sphere)
    pl.subplot(0, 1)
    pl.add_mesh(sphere)
    pl.add_camera3d_widget()
    pl.show(cpos=pl.camera_position)


@pytest.mark.parametrize('outline_opacity', [True, False, np.random.default_rng(0).random()])
def test_outline_opacity(uniform, outline_opacity):
    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    plane_widget = pl.add_plane_widget(
        callback=func, implicit=True, outline_opacity=outline_opacity
    )
    assert plane_widget.GetOutlineProperty().GetOpacity() == float(outline_opacity)
    pl.close()


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_box_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_box_widget(None)
    pl.clear_box_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_plane_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_plane_widget(None)
    pl.clear_plane_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_line_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_line_widget(None)
    pl.clear_line_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_slider_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_slider_widget(None, [0, 1])
    pl.clear_slider_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_spline_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_spline_widget(None)
    pl.clear_spline_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_measure_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_measurement_widget(None)
    pl.clear_measure_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_sphere_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_sphere_widget(None)
    pl.clear_sphere_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_camera_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera_orientation_widget()
    pl.clear_camera_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_button_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_checkbox_button_widget(None)
    pl.clear_button_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_logo_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_logo_widget(None)
    pl.clear_logo_widgets()
    pl.show(cpos='xy')


@pytest.mark.needs_vtk_version(9, 3, 0)
@pytest.mark.usefixtures('verify_image_cache')
def test_clear_camera3d_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera3d_widget()
    pl.clear_camera3d_widgets()
    pl.show(cpos='xy')


class TestEventParser:
    """Class to regroup tests for widgets that use the  `_parse_interaction_event()` function"""

    @pytest.fixture
    def plotter(self):
        yield (pl := pv.Plotter())
        pl.close()

    @pytest.mark.parametrize(
        ('method', 'widget'),
        [
            ('add_box_widget', 'vtkBoxWidget'),
            ('add_plane_widget', 'vtkImplicitPlaneWidget'),
            ('add_line_widget', 'vtkLineWidget'),
            ('add_text_slider_widget', 'vtkSliderWidget'),
            ('add_slider_widget', 'vtkSliderWidget'),
            ('add_sphere_widget', 'vtkSphereWidget'),
            ('add_spline_widget', 'vtkSplineWidget'),
        ],
    )
    def test_add_widget(
        self,
        plotter: pv.Plotter,
        method: str,
        widget: str,
        mocker: MockerFixture,
    ):
        # Arrange
        mock = mocker.patch.object(widgets, '_parse_interaction_event')
        mock_vtk = mocker.patch.object(widgets, '_vtk')

        if widget == 'vtkSplineWidget':
            mocker.patch.object(widgets.pv, 'wrap').return_value = pv.PolyData()

        kwargs = dict(callback=lambda *b: b, interaction_event=(e := 'foo'))
        if widget == 'vtkLineWidget':
            mock_vtk.vtkLineWidget().GetPoint1.return_value = (0,) * 3
            mock_vtk.vtkLineWidget().GetPoint2.return_value = (0,) * 3

        elif widget == 'vtkSliderWidget':
            k = 'data' if method == 'add_text_slider_widget' else 'rng'
            kwargs[k] = [0, 1]

        # Act
        method = getattr(plotter, method)
        method(**kwargs)

        # Assert
        mock.assert_called_with(e)
        getattr(mock_vtk, widget)().AddObserver.assert_called_with(mock(e), ANY)
