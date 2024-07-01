from __future__ import annotations

import re

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.utilities.arrays import vtkmatrix_from_array


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
        "  X label:                    'X'",
        "  Y label:                    'Y'",
        "  Z label:                    'Z'",
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
        "  Scale:                      (1.0, 1.0, 1.0)",
        "  User matrix:                Identity",
        "  Visible:                    True",
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


# def test_origin(axes):
#     origin = np.random.default_rng().random(3)
#     axes.origin = origin
#     assert np.all(axes.GetOrigin() == origin)
#     assert np.all(axes.origin == origin)


def test_axes_assembly_label_position(axes_assembly):
    assert axes_assembly.label_position == (0.8, 0.8, 0.8)
    axes_assembly.label_position = (1, 2, 3)
    assert axes_assembly.label_position == (1, 2, 3)


def test_axes_assembly_label_position_init():
    axes_assembly = pv.AxesAssembly(label_position=2)
    assert axes_assembly.label_position == (2, 2, 2)


def test_axes_assembly_labels(axes_assembly):
    assert axes_assembly.labels == ('X', 'Y', 'Z')
    axes_assembly.labels = ('i', 'j', 'k')
    assert axes_assembly.labels == ('i', 'j', 'k')


def test_axes_assembly_labels_init():
    axes_assembly = pv.AxesAssembly(labels=('i', 'j', 'k'))
    assert axes_assembly.labels == ('i', 'j', 'k')


def test_axes_assembly_x_label(axes_assembly):
    assert axes_assembly.x_label == 'X'
    axes_assembly.x_label = 'label'
    assert axes_assembly.x_label == 'label'


def test_axes_assembly_x_label_init(axes_assembly):
    axes_assembly = pv.AxesAssembly(x_label='label')
    assert axes_assembly.x_label == 'label'


def test_axes_assembly_y_label(axes_assembly):
    assert axes_assembly.y_label == 'Y'
    axes_assembly.y_label = 'label'
    assert axes_assembly.y_label == 'label'


def test_axes_assembly_y_label_init(axes_assembly):
    axes_assembly = pv.AxesAssembly(y_label='label')
    assert axes_assembly.y_label == 'label'


def test_axes_assembly_z_label(axes_assembly):
    assert axes_assembly.z_label == 'Z'
    axes_assembly.z_label = 'label'
    assert axes_assembly.z_label == 'label'


def test_axes_assembly_z_label_init(axes_assembly):
    axes_assembly = pv.AxesAssembly(z_label='label')
    assert axes_assembly.z_label == 'label'


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
    axes_assembly.label_size = 100
    assert axes_assembly.label_size == 100


def test_axes_assembly_label_size_init():
    axes_assembly = pv.AxesAssembly(label_size=42)
    assert axes_assembly.label_size == 42


# def test_axes_assembly_properties(axes_assembly):
#     axes_assembly.x_shaft_prop.ambient = 0.1
#     assert axes_assembly.x_shaft_prop.ambient == 0.1
#     axes_assembly.x_tip_prop.ambient = 0.11
#     assert axes_assembly.x_tip_prop.ambient == 0.11
#
#     axes_assembly.y_shaft_prop.ambient = 0.2
#     assert axes_assembly.y_shaft_prop.ambient == 0.2
#     axes_assembly.y_tip_prop.ambient = 0.12
#     assert axes_assembly.y_tip_prop.ambient == 0.12
#
#     axes_assembly.z_shaft_prop.ambient = 0.3
#     assert axes_assembly.z_shaft_prop.ambient == 0.3
#     axes_assembly.z_tip_prop.ambient = 0.13
#     assert axes_assembly.z_tip_prop.ambient == 0.13
#
#     # Test init
#     prop = pv.Property(ambient=0.42)
#     axes_assembly = pv.AxesAssembly(properties=prop)
#     assert axes_assembly.x_shaft_prop.ambient == 0.42
#     assert axes_assembly.x_shaft_prop is not prop
#     assert axes_assembly.x_tip_prop.ambient == 0.42
#     assert axes_assembly.y_shaft_prop is not prop
#
#     assert axes_assembly.y_shaft_prop.ambient == 0.42
#     assert axes_assembly.y_shaft_prop is not prop
#     assert axes_assembly.y_tip_prop.ambient == 0.42
#     assert axes_assembly.y_shaft_prop is not prop
#
#     assert axes_assembly.z_shaft_prop.ambient == 0.42
#     assert axes_assembly.z_shaft_prop is not prop
#     assert axes_assembly.z_tip_prop.ambient == 0.42
#     assert axes_assembly.z_shaft_prop is not prop
#
#     msg = '`properties` must be a property object or a dictionary.'
#     with pytest.raises(TypeError, match=msg):
#         pv.pv.AxesAssembly(properties="not_a_dict")


# def test_axes_assembly_user_matrix():
#     eye = np.eye(4)
#     eye2 = eye * 2
#     eye3 = eye * 3
#
#     a = pv.AxesActor(_make_orientable=False)
#     assert np.array_equal(a.user_matrix, eye)
#     assert np.array_equal(a._user_matrix, eye)
#     assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye)
#
#     a.user_matrix = eye2
#     assert np.array_equal(a.user_matrix, eye2)
#     assert np.array_equal(a._user_matrix, eye2)
#     assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye2)
#
#     a._make_orientable = True
#     a.SetUserMatrix(vtkmatrix_from_array(eye3))
#     assert np.array_equal(a.user_matrix, eye3)
#     assert np.array_equal(a._user_matrix, eye3)
#     assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye3)


def _compute_expected_bounds(axes_assembly):
    # Compute expected bounds by transforming vtkAxesActor actors

    # Create two vtkAxesActor (one for (+) and one for (-) axes)
    # and store their actors
    actors = []
    vtk_axes = [vtk.vtkAxesActor(), vtk.vtkAxesActor()]
    for axes in vtk_axes:
        axes.SetNormalizedShaftLength(axes_assembly.shaft_length)
        axes.SetNormalizedTipLength(axes_assembly.tip_length)
        axes.SetTotalLength(axes_assembly.total_length)
        axes.SetShaftTypeToCylinder()

        props = vtk.vtkPropCollection()
        axes.GetActors(props)
        actors.extend([props.GetItemAsObject(num) for num in range(6)])

    # Transform actors and get their bounds
    # Half of the actors are reflected to give symmetric axes
    bounds = []
    matrix = axes_assembly._concatenate_implicit_matrix_and_user_matrix()
    matrix_reflect = matrix @ np.diag((-1, -1, -1, 1))
    for i, actor in enumerate(actors):
        m = matrix if i < 6 else matrix_reflect
        actor.SetUserMatrix(vtkmatrix_from_array(m))
        bounds.append(actor.GetBounds())

    b = np.array(bounds)
    return (
        np.min(b[:, 0]),
        np.max(b[:, 1]),
        np.min(b[:, 2]),
        np.max(b[:, 3]),
        np.min(b[:, 4]),
        np.max(b[:, 5]),
    )


# def test_axes_assembly_center(axes_assembly):
#     assert axes_assembly.center == (0, 0, 0)
#     assert axes_assembly.GetCenter() == (0, 0, 0)
#
#     axes_assembly.position = (1, 2, 3)
#     assert axes_assembly.center == axes_assembly.position
#     assert axes_assembly.GetCenter() == axes_assembly.position
#
#     # test center is always the origin when workaround is disabled
#     axes_assembly._make_orientable = False
#     assert axes_assembly.GetCenter() == (0, 0, 0)


# @pytest.mark.parametrize('use_scale', [True, False])
# def test_axes_assembly_length(use_scale):
#     axes_assembly = AxesActorProp()
#     default_length = 3.4641016151377544
#     assert np.allclose(axes_assembly.length, default_length)
#     assert np.allclose(axes_assembly.GetLength(), default_length)
#
#     scaled_length = 7.4833147735478835
#     if use_scale:
#         axes_assembly.scale = (1, 2, 3)
#     else:
#         axes_assembly.total_length = (1, 2, 3)
#     assert np.allclose(axes_assembly.length, scaled_length)
#     assert np.allclose(axes_assembly.GetLength(), scaled_length)
#
#     axes_assembly._make_orientable = False
#     if use_scale:
#         # test length is not correct when workaround is disabled
#         assert np.allclose(axes_assembly.length, default_length)
#         assert np.allclose(axes_assembly.GetLength(), default_length)
#     else:
#         assert np.allclose(axes_assembly.length, scaled_length)
#         assert np.allclose(axes_assembly.GetLength(), scaled_length)


# def test_axes_assembly_symmetric_bounds():
#     axes_assembly = AxesActorProp()
#     default_bounds = (-1, 1, -1, 1, -1, 1)
#     default_length = 3.4641016151377544
#     assert np.allclose(axes_assembly.center, (0, 0, 0))
#     assert np.allclose(axes_assembly.length, default_length)
#     assert np.allclose(axes_assembly.bounds, default_bounds)
#
#     # test radius > length
#     axes_assembly.shaft_radius = 2
#     assert np.allclose(axes_assembly.center, (0, 0, 0))
#     assert np.allclose(axes_assembly.length, 5.542562584220408)
#     assert np.allclose(axes_assembly.bounds, (-1.6, 1.6, -1.6, 1.6, -1.6, 1.6))
#
#     axes_assembly.symmetric_bounds = False
#
#     # make axes geometry tiny to approximate lines and points
#     axes_assembly.shaft_radius = 1e-8
#     axes_assembly.tip_radius = 1e-8
#     axes_assembly.shaft_length = 1
#     axes_assembly.tip_length = 0
#
#     assert np.allclose(axes_assembly.center, (0.5, 0.5, 0.5))
#     assert np.allclose(axes_assembly.length, default_length / 2)
#     assert np.allclose(axes_assembly.bounds, (0, 1, 0, 1, 0, 1))