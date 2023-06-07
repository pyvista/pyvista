"""Test the CubeAxesActor wrapping."""
import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def cube_axes_actor():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl.show_bounds()


def test_cube_axes_actor():
    pl = pv.Plotter()
    actor = pv.CubeAxesActor(
        pl.camera,
        x_label_format=None,
        y_label_format=None,
        z_label_format=None,
    )
    assert isinstance(actor.camera, pv.Camera)

    # ensure label format is set to default
    assert actor.x_label_format == '%.1f'
    assert actor.y_label_format == '%.1f'
    assert actor.z_label_format == '%.1f'


def test_labels(cube_axes_actor):
    # test setting "format" to just a string
    cube_axes_actor.x_label_format = 'Value'
    assert len(cube_axes_actor.x_labels) == 5
    assert set(cube_axes_actor.x_labels) == {'Value'}

    # test no format, values should match exactly
    cube_axes_actor.y_label_format = ''
    assert len(cube_axes_actor.y_labels) == 5
    values = np.array(cube_axes_actor.y_labels, float)
    expected = np.linspace(cube_axes_actor.bounds[2], cube_axes_actor.bounds[3], 5)
    assert np.allclose(values, expected)

    # standard format
    cube_axes_actor.z_label_format = '%.1f'
    assert len(cube_axes_actor.z_labels) == 5
    assert all([len(label) < 5 for label in cube_axes_actor.z_labels])


def test_tick_location(cube_axes_actor):
    assert isinstance(cube_axes_actor.tick_location, str)

    for location in ['inside', 'outside', 'both']:
        cube_axes_actor.tick_location = location
        assert cube_axes_actor.tick_location == location


def test_use_2d_mode(cube_axes_actor):
    assert isinstance(cube_axes_actor.use_2d_mode, bool)
    cube_axes_actor.use_2d_mode = False
    assert cube_axes_actor.use_2d_mode is False


def test_label_visibility_setter(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_label_visibility, bool)
    cube_axes_actor.x_label_visibility = False
    assert cube_axes_actor.x_label_visibility is False

    assert isinstance(cube_axes_actor.y_label_visibility, bool)
    cube_axes_actor.y_label_visibility = False
    assert cube_axes_actor.y_label_visibility is False

    assert isinstance(cube_axes_actor.z_label_visibility, bool)
    cube_axes_actor.z_label_visibility = False
    assert cube_axes_actor.z_label_visibility is False


def test_titles(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_title, str)
    cube_axes_actor.x_title = 'x foo'
    assert cube_axes_actor.x_title == 'x foo'

    assert isinstance(cube_axes_actor.y_title, str)
    cube_axes_actor.y_title = 'y foo'
    assert cube_axes_actor.y_title == 'y foo'

    assert isinstance(cube_axes_actor.z_title, str)
    cube_axes_actor.z_title = 'z foo'
    assert cube_axes_actor.z_title == 'z foo'


def test_axis_minor_tick_visibility(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_axis_minor_tick_visibility, bool)
    cube_axes_actor.x_axis_minor_tick_visibility = False
    assert cube_axes_actor.x_axis_minor_tick_visibility is False

    assert isinstance(cube_axes_actor.y_axis_minor_tick_visibility, bool)
    cube_axes_actor.y_axis_minor_tick_visibility = False
    assert cube_axes_actor.y_axis_minor_tick_visibility is False

    assert isinstance(cube_axes_actor.z_axis_minor_tick_visibility, bool)
    cube_axes_actor.z_axis_minor_tick_visibility = False
    assert cube_axes_actor.z_axis_minor_tick_visibility is False


def test_offset(cube_axes_actor):
    assert isinstance(cube_axes_actor.title_offset, float)
    cube_axes_actor.title_offset = 0.01
    assert cube_axes_actor.title_offset == 0.01

    assert isinstance(cube_axes_actor.label_offset, float)
    cube_axes_actor.label_offset = 0.01
    assert cube_axes_actor.label_offset == 0.01
