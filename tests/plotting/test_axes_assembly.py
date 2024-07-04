from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv


@pytest.fixture()
def axes_assembly():
    return pv.AxesAssembly()


def test_axes_assembly_repr(axes_assembly):
    repr_ = repr(axes_assembly)
    actual_lines = repr_.splitlines()[1:]
    expected_lines = [
        "  Shaft type:                 'cylinder'",
        "  Shaft radius:               0.025",
        "  Shaft length:               (0.8, 0.8, 0.8)",
        "  Tip type:                   'cone'",
        "  Tip radius:                 0.1",
        "  Tip length:                 (0.2, 0.2, 0.2)",
        "  Symmetric:                  False",
        "  Symmetric bounds:           False",
        "  X label:                    'X'",
        "  Y label:                    'Y'",
        "  Z label:                    'Z'",
        "  Label color:                Color(name='black', hex='#000000ff', opacity=255)",
        "  Show labels:                True",
        "  Label position:             (0.8, 0.8, 0.8)",
        "  X Color:                                     ",
        "      Shaft                   Color(name='tomato', hex='#ff6347ff', opacity=255)",
        "      Tip                     Color(name='tomato', hex='#ff6347ff', opacity=255)",
        "  Y Color:                                     ",
        "      Shaft                   Color(name='seagreen', hex='#2e8b57ff', opacity=255)",
        "      Tip                     Color(name='seagreen', hex='#2e8b57ff', opacity=255)",
        "  Z Color:                                     ",
        "      Shaft                   Color(name='mediumblue', hex='#0000cdff', opacity=255)",
        "      Tip                     Color(name='mediumblue', hex='#0000cdff', opacity=255)",
        "  Position:                   (0.0, 0.0, 0.0)",
        "  Orientation:                (0.0, -0.0, 0.0)",
        "  Origin:                     (0.0, 0.0, 0.0)",
        "  Scale:                      (1.0, 1.0, 1.0)",
        "  User matrix:                Identity",
        "  X Bounds                    -1.000E-01, 1.000E+00",
        "  Y Bounds                    -1.000E-01, 1.000E+00",
        "  Z Bounds                    -1.000E-01, 1.000E+00",
    ]
    assert len(actual_lines) == len(expected_lines)
    assert actual_lines == expected_lines

    axes_assembly.user_matrix = np.eye(4) * 2
    repr_ = repr(axes_assembly)
    assert "User matrix:                Set" in repr_


def test_axes_assembly_x_color(axes_assembly):
    axes_assembly.x_color = 'black'
    assert axes_assembly.x_color[0].name == 'black'
    assert axes_assembly._shaft_actors[0].prop.color.name == 'black'

    assert axes_assembly.x_color[1].name == 'black'
    assert axes_assembly._tip_actors[0].prop.color.name == 'black'


def test_axes_assembly_y_color(axes_assembly):
    axes_assembly.y_color = 'black'
    assert axes_assembly.y_color[0].name == 'black'
    assert axes_assembly._shaft_actors[1].prop.color.name == 'black'

    assert axes_assembly.y_color[1].name == 'black'
    assert axes_assembly._tip_actors[1].prop.color.name == 'black'


def test_axes_assembly_z_color(axes_assembly):
    axes_assembly.z_color = 'black'
    assert axes_assembly.z_color[0].name == 'black'
    assert axes_assembly._shaft_actors[2].prop.color.name == 'black'

    assert axes_assembly.z_color[1].name == 'black'
    assert axes_assembly._tip_actors[2].prop.color.name == 'black'


def test_axes_assembly_color_inputs(axes_assembly):
    axes_assembly.x_color = [[255, 255, 255, 255]]
    assert axes_assembly.x_color[0].name == 'white'
    assert axes_assembly.x_color[1].name == 'white'

    axes_assembly.x_color = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert axes_assembly.x_color[0].name == 'red'
    assert axes_assembly.x_color[1].name == 'black'

    err_msg = '\nInput must be a single ColorLike color or a sequence of 2 ColorLike colors.'

    match = 'Invalid color(s):\n\tham'
    with pytest.raises(ValueError, match=re.escape(match + err_msg)):
        axes_assembly.y_color = 'ham'

    match = "Invalid color(s):\n\t['eggs']"
    with pytest.raises(ValueError, match=re.escape(match + err_msg)):
        axes_assembly.y_color = ['eggs']

    match = "Invalid color(s):\n\t['red', 'green', 'blue']"
    with pytest.raises(ValueError, match=re.escape(match + err_msg)):
        axes_assembly.z_color = ['red', 'green', 'blue']


@pytest.fixture()
def _config_axes_theme():
    # Store values
    x_color = pv.global_theme.axes.x_color
    y_color = pv.global_theme.axes.y_color
    z_color = pv.global_theme.axes.z_color
    yield
    # Restore values
    pv.global_theme.axes.x_color = x_color
    pv.global_theme.axes.y_color = y_color
    pv.global_theme.axes.z_color = z_color


@pytest.mark.usefixtures('_config_axes_theme')
def test_axes_assembly_theme(axes_assembly):
    assert axes_assembly.x_color[0].name == 'tomato'
    assert axes_assembly.x_color[1].name == 'tomato'
    assert axes_assembly.y_color[0].name == 'seagreen'
    assert axes_assembly.y_color[1].name == 'seagreen'
    assert axes_assembly.z_color[0].name == 'mediumblue'
    assert axes_assembly.z_color[1].name == 'mediumblue'

    pv.global_theme.axes.x_color = 'black'
    pv.global_theme.axes.y_color = 'white'
    pv.global_theme.axes.z_color = 'gray'

    axes_geometry_source = pv.AxesAssembly()
    assert axes_geometry_source.x_color[0].name == 'black'
    assert axes_geometry_source.x_color[1].name == 'black'
    assert axes_geometry_source.y_color[0].name == 'white'
    assert axes_geometry_source.y_color[1].name == 'white'
    assert axes_geometry_source.z_color[0].name == 'gray'
    assert axes_geometry_source.z_color[1].name == 'gray'


def test_axes_assembly_label_position(axes_assembly):
    assert axes_assembly.label_position == (0.8, 0.8, 0.8)
    label_position = (1, 2, 3)
    axes_assembly.label_position = label_position
    assert axes_assembly.label_position == label_position


def test_axes_assembly_label_position_init():
    label_position = 2
    axes_assembly = pv.AxesAssembly(label_position=label_position)
    assert axes_assembly.label_position == (label_position, label_position, label_position)


def test_axes_assembly_labels(axes_assembly):
    assert axes_assembly.labels == ('X', 'Y', 'Z')
    labels = ('i', 'j', 'k')
    axes_assembly.labels = labels
    assert axes_assembly.labels == labels


def test_axes_assembly_labels_init():
    labels = ('i', 'j', 'k')
    axes_assembly = pv.AxesAssembly(labels=labels)
    assert axes_assembly.labels == labels


def test_axes_assembly_x_label(axes_assembly):
    assert axes_assembly.x_label == 'X'
    x_label = 'label'
    axes_assembly.x_label = x_label
    assert axes_assembly.x_label == x_label


def test_axes_assembly_x_label_init(axes_assembly):
    x_label = 'label'
    axes_assembly = pv.AxesAssembly(x_label=x_label)
    assert axes_assembly.x_label == x_label


def test_axes_assembly_y_label(axes_assembly):
    assert axes_assembly.y_label == 'Y'
    y_label = 'label'
    axes_assembly.y_label = y_label
    assert axes_assembly.y_label == y_label


def test_axes_assembly_y_label_init(axes_assembly):
    y_label = 'label'
    axes_assembly = pv.AxesAssembly(y_label=y_label)
    assert axes_assembly.y_label == y_label


def test_axes_assembly_z_label(axes_assembly):
    assert axes_assembly.z_label == 'Z'
    z_label = 'label'
    axes_assembly.z_label = z_label
    assert axes_assembly.z_label == z_label


def test_axes_assembly_z_label_init(axes_assembly):
    z_label = 'label'
    axes_assembly = pv.AxesAssembly(z_label=z_label)
    assert axes_assembly.z_label == z_label


def test_axes_assembly_labels_raises():
    match = "Cannot initialize '{}' and 'labels' properties together. Specify one or the other, not both."
    with pytest.raises(ValueError, match=match.format('x_label')):
        pv.AxesAssembly(x_label='A', y_label='B', z_label='C', labels='UVW')
    with pytest.raises(ValueError, match=match.format('y_label')):
        pv.AxesAssembly(y_label='B', z_label='C', labels='UVW')
    with pytest.raises(ValueError, match=match.format('z_label')):
        pv.AxesAssembly(z_label='C', labels='UVW')


def test_axes_assembly_show_labels(axes_assembly):
    assert axes_assembly.show_labels is True
    axes_assembly.show_labels = False
    assert axes_assembly.show_labels is False


def test_axes_assembly_show_labels_init():
    axes_assembly = pv.AxesAssembly(show_labels=False)
    assert axes_assembly.show_labels is False


def test_axes_assembly_label_size(axes_assembly):
    assert axes_assembly.label_size == 50
    label_size = 100
    axes_assembly.label_size = label_size
    assert axes_assembly.label_size == label_size


def test_axes_assembly_label_size_init():
    label_size = 42
    axes_assembly = pv.AxesAssembly(label_size=label_size)
    assert axes_assembly.label_size == label_size


def test_axes_assembly_origin(axes_assembly):
    assert axes_assembly.origin == (0, 0, 0)
    origin = (1, 2, 3)
    axes_assembly.origin = origin
    assert axes_assembly.origin == origin


def test_axes_assembly_origin_init():
    origin = (1, 2, 3)
    axes_assembly = pv.AxesAssembly(origin=origin)
    assert axes_assembly.origin == origin


def test_axes_assembly_scale(axes_assembly):
    assert axes_assembly.scale == (1.0, 1.0, 1.0)
    scale = (1, 2, 3)
    axes_assembly.scale = scale
    assert axes_assembly.scale == scale


def test_axes_assembly_scale_init():
    scale = (1, 2, 3)
    axes_assembly = pv.AxesAssembly(scale=scale)
    assert axes_assembly.scale == scale


def test_axes_assembly_position(axes_assembly):
    assert axes_assembly.position == (0, 0, 0)
    position = (1, 2, 3)
    axes_assembly.position = position
    assert axes_assembly.position == position


def test_axes_assembly_position_init():
    position = (1, 2, 3)
    axes_assembly = pv.AxesAssembly(position=position)
    assert axes_assembly.position == position


def test_axes_assembly_orientation(axes_assembly):
    assert axes_assembly.orientation == (0, 0, 0)
    orientation = (1, 2, 4)
    axes_assembly.orientation = orientation
    assert axes_assembly.orientation == orientation


def test_axes_assembly_orientation_init():
    orientation = (1, 2, 4)
    axes_assembly = pv.AxesAssembly(orientation=orientation)
    assert axes_assembly.orientation == orientation


def test_axes_assembly_user_matrix(axes_assembly):
    assert np.array_equal(axes_assembly.user_matrix, np.eye(4))
    user_matrix = np.diag((2, 3, 4, 1))
    axes_assembly.user_matrix = user_matrix
    assert np.array_equal(axes_assembly.user_matrix, user_matrix)


def test_axes_assembly_user_matrix_init():
    user_matrix = np.diag((2, 3, 4, 1))
    axes_assembly = pv.AxesAssembly(user_matrix=user_matrix)
    assert np.array_equal(axes_assembly.user_matrix, user_matrix)


def test_axes_assembly_center(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert axes_assembly.center == tuple(dataset.center)


def test_axes_assembly_bounds(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert axes_assembly.bounds == tuple(dataset.bounds)


def test_axes_assembly_length(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert axes_assembly.length == dataset.length
