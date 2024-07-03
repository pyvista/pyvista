from __future__ import annotations

import re

import pytest

import pyvista as pv


@pytest.fixture()
def axes_assembly():
    return pv.AxesAssembly()


@pytest.fixture()
def symmetric_axes_assembly():
    return pv.SymmetricAxesAssembly()


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
    ]
    assert len(actual_lines) == len(expected_lines)
    assert actual_lines == expected_lines


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


def test_symmetric_axes_assembly(symmetric_axes_assembly):
    assert symmetric_axes_assembly.GetBounds() == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_symmetric_axes_assembly_set_get_labels(symmetric_axes_assembly):
    labels = 'A', 'B', 'C'
    symmetric_axes_assembly.labels = labels
    assert symmetric_axes_assembly.labels == ('+A', '-A', '+B', '-B', '+C', '-C')

    labels = ('1', '2', '3', '4', '5', '6')
    symmetric_axes_assembly.labels = labels
    assert symmetric_axes_assembly.labels == labels


@pytest.mark.parametrize('test_property', ['x_label', 'y_label', 'z_label'])
def test_symmetric_axes_assembly_set_get_label(symmetric_axes_assembly, test_property):
    label = 'U'
    setattr(symmetric_axes_assembly, test_property, label)
    assert getattr(symmetric_axes_assembly, test_property) == ('+' + label, '-' + label)

    label = ('plus', 'minus')
    setattr(symmetric_axes_assembly, test_property, label)
    assert getattr(symmetric_axes_assembly, test_property) == label


@pytest.mark.parametrize('test_property', ['x_label', 'y_label', 'z_label'])
def test_symmetric_axes_assembly_init_label(test_property):
    label = 'U'
    kwargs = {test_property: label}
    axes_assembly = pv.SymmetricAxesAssembly(**kwargs)
    assert getattr(axes_assembly, test_property) == ('+' + label, '-' + label)

    label = ('plus', 'minus')
    kwargs = {test_property: label}
    axes_assembly = pv.SymmetricAxesAssembly(**kwargs)
    assert getattr(axes_assembly, test_property) == label
