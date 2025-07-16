from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def axes_assembly():
    return pv.AxesAssembly()


@pytest.fixture
def axes_assembly_symmetric():
    return pv.AxesAssemblySymmetric()


@pytest.fixture
def planes_assembly():
    return pv.PlanesAssembly()


def test_axes_assembly_repr(axes_assembly):
    repr_ = repr(axes_assembly)
    actual_lines = repr_.splitlines()[1:]
    expected_lines = [
        "  Shaft type:                 'cylinder'",
        '  Shaft radius:               0.025',
        '  Shaft length:               (0.8, 0.8, 0.8)',
        "  Tip type:                   'cone'",
        '  Tip radius:                 0.1',
        '  Tip length:                 (0.2, 0.2, 0.2)',
        '  Symmetric:                  False',
        '  Symmetric bounds:           False',
        "  X label:                    'X'",
        "  Y label:                    'Y'",
        "  Z label:                    'Z'",
        "  Label color:                Color(name='black', hex='#000000ff', opacity=255)",
        '  Show labels:                True',
        '  Label position:             (0.8, 0.8, 0.8)',
        '  X Color:                                     ',
        "      Shaft                   Color(name='tomato', hex='#ff6347ff', opacity=255)",
        "      Tip                     Color(name='tomato', hex='#ff6347ff', opacity=255)",
        '  Y Color:                                     ',
        "      Shaft                   Color(name='seagreen', hex='#2e8b57ff', opacity=255)",
        "      Tip                     Color(name='seagreen', hex='#2e8b57ff', opacity=255)",
        '  Z Color:                                     ',
        "      Shaft                   Color(name='mediumblue', hex='#0000cdff', opacity=255)",
        "      Tip                     Color(name='mediumblue', hex='#0000cdff', opacity=255)",
        '  Position:                   (0.0, 0.0, 0.0)',
        '  Orientation:                (0.0, -0.0, 0.0)',
        '  Origin:                     (0.0, 0.0, 0.0)',
        '  Scale:                      (1.0, 1.0, 1.0)',
        '  User matrix:                Identity',
        '  X Bounds                    -1.000E-01, 1.000E+00',
        '  Y Bounds                    -1.000E-01, 1.000E+00',
        '  Z Bounds                    -1.000E-01, 1.000E+00',
    ]
    assert len(actual_lines) == len(expected_lines)
    assert actual_lines == expected_lines

    axes_assembly.user_matrix = np.eye(4) * 2
    repr_ = repr(axes_assembly)
    assert 'User matrix:                Set' in repr_


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


@pytest.fixture
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
    assert np.allclose(axes_assembly.label_position, (0.8, 0.8, 0.8))
    label_position = (1.0, 2.0, 3.0)
    axes_assembly.label_position = label_position
    assert np.allclose(axes_assembly.label_position, label_position)


def test_axes_assembly_label_position_init():
    label_position = 2.0
    axes_assembly = pv.AxesAssembly(label_position=label_position)
    assert np.allclose(
        axes_assembly.label_position, (label_position, label_position, label_position)
    )


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
    match = (
        "Cannot initialize '{}' and 'labels' properties together. "
        'Specify one or the other, not both.'
    )
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
    assert np.allclose(axes_assembly.origin, (0.0, 0.0, 0.0))
    origin = (1.0, 2.0, 3.0)
    axes_assembly.origin = origin
    assert np.allclose(axes_assembly.origin, origin)


def test_axes_assembly_origin_init():
    origin = (1.0, 2.0, 3.0)
    axes_assembly = pv.AxesAssembly(origin=origin)
    assert np.allclose(axes_assembly.origin, origin)


def test_axes_assembly_scale(axes_assembly):
    assert np.allclose(axes_assembly.scale, (1.0, 1.0, 1.0))
    scale = (1.0, 2.0, 3.0)
    axes_assembly.scale = scale
    assert np.allclose(axes_assembly.scale, scale)


def test_axes_assembly_scale_init():
    scale = (1.0, 2.0, 3.0)
    axes_assembly = pv.AxesAssembly(scale=scale)
    assert np.allclose(axes_assembly.scale, scale)


def test_axes_assembly_position(axes_assembly):
    assert np.allclose(axes_assembly.position, (0.0, 0.0, 0.0))
    position = (1.0, 2.0, 3.0)
    axes_assembly.position = position
    assert np.allclose(axes_assembly.position, position)


def test_axes_assembly_position_init():
    position = (1.0, 2.0, 3.0)
    axes_assembly = pv.AxesAssembly(position=position)
    assert np.allclose(axes_assembly.position, position)


def test_axes_assembly_orientation(axes_assembly):
    assert np.allclose(axes_assembly.orientation, (0.0, 0.0, 0.0))
    orientation = (1.0, 2.0, 4.0)
    axes_assembly.orientation = orientation
    assert np.allclose(axes_assembly.orientation, orientation)


def test_axes_assembly_orientation_init():
    orientation = (1.0, 2.0, 4.0)
    axes_assembly = pv.AxesAssembly(orientation=orientation)
    assert np.allclose(axes_assembly.orientation, orientation)


def test_axes_assembly_user_matrix(axes_assembly):
    assert np.allclose(axes_assembly.user_matrix, np.eye(4))
    user_matrix = np.diag((2.0, 3.0, 4.0, 1.0))
    axes_assembly.user_matrix = user_matrix
    assert np.allclose(axes_assembly.user_matrix, user_matrix)


def test_axes_assembly_user_matrix_init():
    user_matrix = np.diag((2.0, 3.0, 4.0, 1.0))
    axes_assembly = pv.AxesAssembly(user_matrix=user_matrix)
    assert np.allclose(axes_assembly.user_matrix, user_matrix)


def test_axes_assembly_center(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert np.allclose(axes_assembly.center, dataset.center)


def test_axes_assembly_bounds(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert np.allclose(axes_assembly.bounds, dataset.bounds)


def test_axes_assembly_length(axes_assembly):
    # Test param matches value from underlying dataset
    dataset = axes_assembly._shaft_and_tip_geometry_source.output
    assert np.allclose(axes_assembly.length, dataset.length)


def test_axes_assembly_name():
    axes = pv.PlanesAssembly(name='axes')
    assert axes.name == 'axes'


def test_axes_assembly_symmetric(axes_assembly_symmetric):
    assert np.allclose(axes_assembly_symmetric.bounds, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))


def test_axes_assembly_symmetric_set_get_labels(axes_assembly_symmetric):
    labels = 'A', 'B', 'C'
    axes_assembly_symmetric.labels = labels
    assert axes_assembly_symmetric.labels == ('+A', '-A', '+B', '-B', '+C', '-C')

    labels = ('1', '2', '3', '4', '5', '6')
    axes_assembly_symmetric.labels = labels
    assert axes_assembly_symmetric.labels == labels


@pytest.mark.parametrize('test_property', ['x_label', 'y_label', 'z_label'])
def test_axes_assembly_symmetric_set_get_label(axes_assembly_symmetric, test_property):
    label = 'U'
    setattr(axes_assembly_symmetric, test_property, label)
    assert getattr(axes_assembly_symmetric, test_property) == ('+' + label, '-' + label)

    label = ('plus', 'minus')
    setattr(axes_assembly_symmetric, test_property, label)
    assert getattr(axes_assembly_symmetric, test_property) == label


@pytest.mark.parametrize('test_property', ['x_label', 'y_label', 'z_label'])
def test_axes_assembly_symmetric_init_label(test_property):
    label = 'U'
    kwargs = {test_property: label}
    axes_assembly = pv.AxesAssemblySymmetric(**kwargs)
    assert getattr(axes_assembly, test_property) == ('+' + label, '-' + label)

    label = ('plus', 'minus')
    kwargs = {test_property: label}
    axes_assembly = pv.AxesAssemblySymmetric(**kwargs)
    assert getattr(axes_assembly, test_property) == label


def test_axes_assembly_symmetric_name():
    axes = pv.PlanesAssembly(name='axes')
    assert axes.name == 'axes'


def test_axes_assembly_set_get_part_prop_all(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 1.0)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    axes_assembly.set_actor_prop('ambient', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))


def test_axes_assembly_set_get_actor_prop_all_color(axes_assembly):
    float_rgb = (1.0, 1.0, 1.0)
    axes_assembly.set_actor_prop('color', float_rgb)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    expected_rgb = [
        float_rgb,
        float_rgb,
        float_rgb,
        float_rgb,
        float_rgb,
        float_rgb,
    ]
    assert np.allclose(actual_rgb, expected_rgb)

    expected_rgb = [
        tuple(np.array(float_rgb) * 10 / 255),
        tuple(np.array(float_rgb) * 20 / 255),
        tuple(np.array(float_rgb) * 30 / 255),
        tuple(np.array(float_rgb) * 40 / 255),
        tuple(np.array(float_rgb) * 50 / 255),
        tuple(np.array(float_rgb) * 60 / 255),
    ]
    axes_assembly.set_actor_prop('color', expected_rgb)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    assert np.allclose(actual_rgb, expected_rgb)


def test_axes_assembly_set_get_actor_prop_axis(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 0.5, axis=0)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.5, 0.0, 0.0, 0.5, 0.0, 0.0))

    axes_assembly.set_actor_prop('ambient', [0.1, 0.2], axis='x')
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val.x_shaft, 0.1)
    assert np.allclose(val.x_tip, 0.2)
    assert np.allclose(val, (0.1, 0.0, 0.0, 0.2, 0.0, 0.0))


def test_axes_assembly_set_get_actor_prop_axis_color(axes_assembly):
    float_rgb = (1.0, 1.0, 1.0)
    axes_assembly.set_actor_prop('color', float_rgb, axis=0)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    expected_rgb = [
        float_rgb,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
        float_rgb,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
    ]
    assert np.allclose(actual_rgb, expected_rgb)

    color1, color2 = (
        tuple(np.array(float_rgb) * 10 / 255),
        tuple(np.array(float_rgb) * 40 / 255),
    )
    axes_assembly.set_actor_prop('color', [color1, color2], axis=0)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    expected_rgb = [
        color1,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
        color2,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
    ]
    assert np.allclose(actual_rgb, expected_rgb)


def test_axes_assembly_set_get_actor_prop_axis_and_tip(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 0.7, axis=1, part=1)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.0, 0.0, 0.7, 0.0))

    axes_assembly.set_actor_prop('ambient', [0.7], axis='y', part='tip')
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.0, 0.0, 0.7, 0.0))


def test_axes_assembly_set_get_actor_prop_axis_and_tip_color(axes_assembly):
    float_rgb = (1.0, 1.0, 1.0)
    axes_assembly.set_actor_prop('color', float_rgb, axis=1, part=1)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    expected_rgb = [
        pv.Color('tomato').float_rgb,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
        pv.Color('tomato').float_rgb,
        float_rgb,
        pv.Color('mediumblue').float_rgb,
    ]
    assert np.allclose(actual_rgb, expected_rgb)

    axes_assembly.set_actor_prop('color', [float_rgb], axis=1, part=1)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    assert np.allclose(actual_rgb, expected_rgb)


def test_axes_assembly_set_get_actor_prop_axis_and_shaft(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 0.1, axis=2, part=0)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.1, 0.0, 0.0, 0.0))

    axes_assembly.set_actor_prop('ambient', [0.1], axis='z', part='shaft')
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.1, 0.0, 0.0, 0.0))


def test_axes_assembly_set_get_actor_prop_shaft(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 0.3, part='shaft')
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.3, 0.3, 0.3, 0.0, 0.0, 0.0))

    axes_assembly.set_actor_prop('ambient', [0.1, 0.2, 0.4], part=0)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.1, 0.2, 0.4, 0.0, 0.0, 0.0))


def test_axes_assembly_set_get_actor_prop_shaft_color(axes_assembly):
    float_rgb = (1.0, 1.0, 1.0)
    axes_assembly.set_actor_prop('color', float_rgb, part=0)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    expected_rgb = [
        float_rgb,
        float_rgb,
        float_rgb,
        pv.Color('tomato').float_rgb,
        pv.Color('seagreen').float_rgb,
        pv.Color('mediumblue').float_rgb,
    ]
    assert np.allclose(actual_rgb, expected_rgb)

    axes_assembly.set_actor_prop('color', [float_rgb], part=0)
    val = axes_assembly.get_actor_prop('color')
    actual_rgb = [color.float_rgb for color in val]
    assert np.allclose(actual_rgb, expected_rgb)


def test_axes_assembly_set_get_actor_prop_tip(axes_assembly):
    axes_assembly.set_actor_prop('ambient', 0.3, part='tip')
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.0, 0.3, 0.3, 0.3))

    axes_assembly.set_actor_prop('ambient', [0.1, 0.2, 0.4], part=1)
    val = axes_assembly.get_actor_prop('ambient')
    assert np.allclose(val, (0.0, 0.0, 0.0, 0.1, 0.2, 0.4))


def test_axes_assembly_set_get_actor_prop_raises(axes_assembly):
    match = "Part must be one of ['shaft', 'tip', 'all']."
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_assembly.set_actor_prop('ambient', 0.0, part=2)

    match = "Axis must be one of ['x', 'y', 'z', 'all']."
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_assembly.set_actor_prop('ambient', 0.0, axis='a')

    match = (
        "Number of values (3) in [1, 2, 3] must match the number of actors (2) for axis '0' "
        "and part 'all'"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_assembly.set_actor_prop('ambient', [1, 2, 3], axis=0)

    match = (
        "Number of values (2) in [0, 1] must match the number of actors (3) for axis 'all' "
        "and part 'shaft'"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_assembly.set_actor_prop('ambient', [0, 1], part='shaft')


def test_planes_assembly_repr(planes_assembly):
    repr_ = repr(planes_assembly)
    actual_lines = repr_.splitlines()[1:]
    expected_lines = [
        '  Resolution:                 (2, 2, 2)',
        "  Normal sign:                ('+', '+', '+')",
        "  X label:                    'YZ'",
        "  Y label:                    'ZX'",
        "  Z label:                    'XY'",
        "  Label color:                Color(name='black', hex='#000000ff', opacity=255)",
        '  Show labels:                True',
        '  Label position:             (0.5, 0.5, 0.5)',
        "  Label edge:                 ('right', 'right', 'right')",
        '  Label offset:               0.05',
        "  Label mode:                 '3D'",
        "  X Color:                    Color(name='tomato', hex='#ff6347ff', opacity=255)",
        "  Y Color:                    Color(name='seagreen', hex='#2e8b57ff', opacity=255)",
        "  Z Color:                    Color(name='mediumblue', hex='#0000cdff', opacity=255)",
        '  Position:                   (0.0, 0.0, 0.0)',
        '  Orientation:                (0.0, -0.0, 0.0)',
        '  Origin:                     (0.0, 0.0, 0.0)',
        '  Scale:                      (1.0, 1.0, 1.0)',
        '  User matrix:                Identity',
        '  X Bounds                    -1.000E+00, 1.000E+00',
        '  Y Bounds                    -1.000E+00, 1.000E+00',
        '  Z Bounds                    -1.000E+00, 1.000E+00',
    ]
    assert len(actual_lines) == len(expected_lines)
    assert actual_lines == expected_lines

    planes_assembly.user_matrix = np.eye(4) * 2
    repr_ = repr(planes_assembly)
    assert 'User matrix:                Set' in repr_


def test_planes_assembly_x_color(planes_assembly):
    planes_assembly.x_color = 'black'
    assert planes_assembly.x_color.name == 'black'


def test_planes_assembly_y_color(planes_assembly):
    planes_assembly.y_color = 'black'
    assert planes_assembly.y_color.name == 'black'


def test_planes_assembly_z_color(planes_assembly):
    planes_assembly.z_color = 'black'
    assert planes_assembly.z_color.name == 'black'


def test_planes_assembly_labels(planes_assembly):
    assert planes_assembly.labels == ('YZ', 'ZX', 'XY')
    labels = ('i', 'j', 'k')
    planes_assembly.labels = labels
    assert planes_assembly.labels == labels


def test_planes_assembly_labels_init():
    labels = ('i', 'j', 'k')
    planes_assembly = pv.PlanesAssembly(labels=labels)
    assert planes_assembly.labels == labels


def test_planes_assembly_x_label(planes_assembly):
    assert planes_assembly.x_label == 'YZ'
    x_label = 'label'
    planes_assembly.x_label = x_label
    assert planes_assembly.x_label == x_label
    assert planes_assembly._planes.get_block_name(0) == x_label


def test_planes_assembly_x_label_init():
    x_label = 'label'
    planes_assembly = pv.PlanesAssembly(x_label=x_label)
    assert planes_assembly.x_label == x_label


def test_planes_assembly_y_label(planes_assembly):
    assert planes_assembly.y_label == 'ZX'
    y_label = 'label'
    planes_assembly.y_label = y_label
    assert planes_assembly.y_label == y_label
    assert planes_assembly._planes.get_block_name(1) == y_label


def test_planes_assembly_y_label_init():
    y_label = 'label'
    planes_assembly = pv.PlanesAssembly(y_label=y_label)
    assert planes_assembly.y_label == y_label


def test_planes_assembly_z_label(planes_assembly):
    assert planes_assembly.z_label == 'XY'
    z_label = 'label'
    planes_assembly.z_label = z_label
    assert planes_assembly.z_label == z_label
    assert planes_assembly._planes.get_block_name(2) == z_label


def test_planes_assembly_z_label_init():
    z_label = 'label'
    planes_assembly = pv.PlanesAssembly(z_label=z_label)
    assert planes_assembly.z_label == z_label


def test_planes_assembly_label_mode(planes_assembly):
    assert planes_assembly.label_mode == '3D'
    label_mode = '2D'
    planes_assembly.label_mode = label_mode
    assert planes_assembly.label_mode == label_mode


def test_planes_assembly_label_mode_init():
    label_mode = '2D'
    planes_assembly = pv.PlanesAssembly(label_mode=label_mode)
    assert planes_assembly.label_mode == label_mode


def test_planes_assembly_opacity(planes_assembly):
    assert np.allclose(planes_assembly.opacity, (0.3, 0.3, 0.3))
    opacity = 0.0, 0.1, 0.2
    planes_assembly.opacity = opacity
    assert np.allclose(planes_assembly.opacity, opacity)


def test_planes_assembly_opacity_init():
    opacity = 0.5
    planes_assembly = pv.PlanesAssembly(opacity=opacity)
    assert np.allclose(planes_assembly.opacity, (opacity, opacity, opacity))


def test_planes_assembly_label_size(planes_assembly):
    assert planes_assembly.label_size == 50
    label_size = 10
    planes_assembly.label_size = label_size
    assert planes_assembly.label_size == label_size


def test_planes_assembly_label_size_init():
    label_size = 10
    planes_assembly = pv.PlanesAssembly(label_size=label_size)
    assert planes_assembly.label_size == label_size


def test_planes_assembly_camera(planes_assembly):
    assert planes_assembly.camera is None

    camera = pv.Camera()
    planes_assembly.camera = camera
    assert planes_assembly.camera is camera


def test_planes_assembly_name():
    planes = pv.PlanesAssembly(name='planes')
    assert planes.name == 'planes'
