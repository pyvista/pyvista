from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.utilities.arrays import vtkmatrix_from_array


@pytest.fixture()
def axes_assembly():
    return pv.AxesAssembly()


# def test_origin(axes):
#     origin = np.random.default_rng().random(3)
#     axes.origin = origin
#     assert np.all(axes.GetOrigin() == origin)
#     assert np.all(axes.origin == origin)
#
#
# def test_axes_symmetric(axes):
#     # test showing
#     assert not axes.GetSymmetric()
#     axes.show_symmetric()
#     assert axes.GetSymmetric()
#
#     # test hiding
#     assert axes.GetSymmetric()
#     axes.hide_symmetric()
#     assert not axes.GetSymmetric()


def test_axes_assembly_label_position(axes_assembly):
    assert axes_assembly.label_position == (1.1, 1.1, 1.1)
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


def test_axes_assembly_label_color(axes_assembly):
    assert axes_assembly.label_color[0].name == 'black'
    assert axes_assembly.label_color[1].name == 'black'
    assert axes_assembly.label_color[2].name == 'black'

    axes_assembly.label_color = 'purple'
    assert len(axes_assembly.label_color) == 3
    assert axes_assembly.label_color[0].name == 'purple'
    assert axes_assembly.label_color[1].name == 'purple'
    assert axes_assembly.label_color[2].name == 'purple'

    axes_assembly.label_color = 'r', 'g', 'b'
    assert len(axes_assembly.label_color) == 3
    assert axes_assembly.label_color[0].name == 'red'
    assert axes_assembly.label_color[1].name == 'green'
    assert axes_assembly.label_color[2].name == 'blue'

    axes_assembly.label_color = ['red', 'green', 'blue']
    assert len(axes_assembly.label_color) == 3
    assert axes_assembly.label_color[0].name == 'red'
    assert axes_assembly.label_color[1].name == 'green'
    assert axes_assembly.label_color[2].name == 'blue'

    axes_assembly.label_color = [1, 2, 3]
    assert len(axes_assembly.label_color) == 3
    assert np.array_equal(axes_assembly.label_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_assembly.label_color[1].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_assembly.label_color[2].int_rgb, [1, 2, 3])


def test_axes_assembly_label_color_init():
    axes_assembly = pv.AxesAssembly(label_color='yellow')
    assert axes_assembly.label_color[0].name == 'yellow'
    assert axes_assembly.label_color[1].name == 'yellow'
    assert axes_assembly.label_color[2].name == 'yellow'


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


rng = np.random.default_rng(42)


# @pytest.mark.parametrize('shaft_tip_length', [(0, 1), (0.2, 0.3), (0.3, 0.8), (0.4, 0.6), (1, 0)])
# @pytest.mark.parametrize('total_length', [[1, 1, 1], [4, 3, 2], [0.4, 0.5, 1.1]])
# @pytest.mark.parametrize('scale', [[1, 1, 1], [0.1, 0.2, 0.3], [2, 3, 4]])
# @pytest.mark.parametrize('position', [[0, 0, 0], [2, 3, 4]])
# def test_axes_assembly_GetBounds(shaft_tip_length, total_length, scale, position):
#     shaft_length, tip_length = shaft_tip_length
#
#     # Test the override for GetBounds() returns the same result as without override
#     # for zero-centered, axis-aligned cases (i.e. no position, scale, etc.)
#     vtk_axes_assembly = vtk.vtkAxesActor()
#     vtk_axes_assembly.SetNormalizedShaftLength(shaft_length, shaft_length, shaft_length)
#     vtk_axes_assembly.SetNormalizedTipLength(tip_length, tip_length, tip_length)
#     vtk_axes_assembly.SetTotalLength(total_length)
#     vtk_axes_assembly.SetShaftTypeToCylinder()
#     expected = vtk_axes_assembly.GetBounds()
#
#     axes_assembly = pv.AxesActor()
#         shaft_length=shaft_length,
#         tip_length=tip_length,
#         total_length=total_length,
#         auto_length=False,
#         _make_orientable=True,
#     )
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly._make_orientable = False
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     # test GetBounds() returns correct output when axes are orientable
#     axes_assembly._make_orientable = True
#     axes_assembly.position = position
#     axes_assembly.scale = scale
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     # test that changing properties dynamically updates the axes
#     axes_assembly.position = rng.random(3) * 2 - 1
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly.scale = rng.random(3) * 2
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly.user_matrix = np.diag(np.append(rng.random(3), 1))
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly.tip_length = rng.random(3)
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly.shaft_length = rng.random(3)
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     axes_assembly.total_length = rng.random(3) * 2
#     expected = _compute_expected_bounds(axes_assembly)
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)
#
#     # test disabling workaround correctly resets axes actors
#     vtk_axes_assembly = vtk.vtkAxesActor()
#     vtk_axes_assembly.SetPosition(axes_assembly.position)
#     vtk_axes_assembly.SetScale(axes_assembly.scale)
#     vtk_axes_assembly.SetNormalizedShaftLength(axes_assembly.shaft_length)
#     vtk_axes_assembly.SetNormalizedTipLength(axes_assembly.tip_length)
#     vtk_axes_assembly.SetTotalLength(axes_assembly.total_length)
#     vtk_axes_assembly.SetShaftTypeToCylinder()
#
#     axes_assembly._make_orientable = False
#     expected = vtk_axes_assembly.GetBounds()
#     actual = axes_assembly.GetBounds()
#     assert np.allclose(actual, expected)


def get_matrix_cases():
    from enum import IntEnum

    class Cases(IntEnum):
        ALL = 0
        ORIGIN = 1
        SCALE = 2
        USER_MATRIX = 3
        ROTATE_X = 4
        ROTATE_Y = 5
        ROTATE_Z = 6
        POSITION = 7
        ORIENTATION = 8

    return Cases


# @pytest.mark.parametrize('case', range(len(get_matrix_cases())))
# def test_axes_assembly_enable_orientation(axes_assembly, vtk_axes_assembly, case):
#     # NOTE: This test works by asserting that:
#     #   all(vtkAxesActor.GetMatrix()) == all(axes_assembly.GetUserMatrix())
#     #
#     # Normally, GetUserMatrix() and GetMatrix() are very different matrices:
#     # - GetUserMatrix() is an independent user-provided transformation matrix
#     # - GetMatrix() concatenates
#     #     implicit_transform -> user_transform -> coordinate system transform
#     # However, as a workaround for pyvista#5019, UserMatrix is used
#     # to represent the user_transform *and* implicit_transform.
#     # Since the renderer's coordinate system transform is identity
#     # by default, this means that in for this test the assertion should
#     # hold true.
#
#     cases = get_matrix_cases()
#     angle = 42
#     origin = (1, 5, 10)
#     scale = (4, 5, 6)
#     orientation = (10, 20, 30)
#     position = (7, 8, 9)
#     user_matrix = vtkmatrix_from_array(
#         [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [0, 0, 0, 1]],
#     )
#
#     # test each property separately and also all together
#     axes_assembly._make_orientable = True
#     if case in [cases.ALL, cases.ORIENTATION]:
#         vtk_axes_assembly.SetOrientation(*orientation)
#         axes_assembly.orientation = orientation
#     if case in [cases.ALL, cases.ORIGIN]:
#         vtk_axes_assembly.SetOrigin(*origin)
#         axes_assembly.origin = origin
#     if case in [cases.ALL, cases.SCALE]:
#         vtk_axes_assembly.SetScale(*scale)
#         axes_assembly.scale = scale
#     if case in [cases.ALL, cases.POSITION]:
#         vtk_axes_assembly.SetPosition(*position)
#         axes_assembly.position = position
#     if case in [cases.ALL, cases.ROTATE_X]:
#         vtk_axes_assembly.RotateX(angle)
#         axes_assembly.rotate_x(angle)
#     if case in [cases.ALL, cases.ROTATE_Y]:
#         vtk_axes_assembly.RotateY(angle * 2)
#         axes_assembly.rotate_y(angle * 2)
#     if case in [cases.ALL, cases.ROTATE_Z]:
#         vtk_axes_assembly.RotateZ(angle * 3)
#         axes_assembly.rotate_z(angle * 3)
#     if case in [cases.ALL, cases.USER_MATRIX]:
#         vtk_axes_assembly.SetUserMatrix(user_matrix)
#         axes_assembly.user_matrix = user_matrix
#
#     expected = array_from_vtkmatrix(vtk_axes_assembly.GetMatrix())
#     actual = array_from_vtkmatrix(axes_assembly._actors[0].GetUserMatrix())
#     assert np.allclose(expected, actual)
#
#     default_bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
#     assert np.allclose(vtk_axes_assembly.GetBounds(), default_bounds)
#
#     # test AxesActor has non-default bounds except in ORIGIN case
#     actual_bounds = axes_assembly.GetBounds()
#     if case == cases.ORIGIN:
#         assert np.allclose(actual_bounds, default_bounds)
#     else:
#         assert not np.allclose(actual_bounds, default_bounds)
#
#     # test that bounds are always default (i.e. incorrect) after disabling orientation
#     axes_assembly._make_orientable = False
#     actual_bounds = axes_assembly.GetBounds()
#     assert np.allclose(actual_bounds, default_bounds)


# def test_axes_assembly_repr(axes_assembly):
#     repr_ = repr(axes_assembly)
#     actual_lines = repr_.splitlines()[1:]
#     expected_lines = [
#         # "  X label:                    'X'",
#         # "  Y label:                    'Y'",
#         # "  Z label:                    'Z'",
#         # "  Show labels:                True",
#         # "  Label position:             (1.0, 1.0, 1.0)",
#         "  Shaft type:                 'cylinder'",
#         "  Shaft radius:               0.05",
#         "  Shaft length:               (0.8, 0.8, 0.8)",
#         "  Tip type:                   'cone'",
#         "  Tip radius:                 0.2",
#         "  Tip length:                 (0.2, 0.2, 0.2)",
#         "  Total length:               (1.0, 1.0, 1.0)",
#         "  Position:                   (0.0, 0.0, 0.0)",
#         "  Scale:                      (1.0, 1.0, 1.0)",
#         "  User matrix:                Identity",
#         "  Visible:                    True",
#         "  X Bounds                    -1.000E-01, 1.000E+00",
#         "  Y Bounds                    -1.000E-01, 1.000E+00",
#         "  Z Bounds                    -1.000E-01, 1.000E+00",
#     ]
#     assert len(actual_lines) == len(expected_lines)
#     assert actual_lines == expected_lines
#
#     axes_assembly.shaft_type = 'cuboid'
#     repr_ = repr(axes_assembly)
#     assert "'cuboid'" in repr_
#
#     # axes_assembly.user_matrix = np.eye(4) * 2
#     # repr_ = repr(axes_assembly)
#     # assert "User matrix:                Set" in repr_


# @pytest.mark.parametrize('use_axis_num', [True, False])
# def test_axes_assembly_set_get_prop(axes_assembly, use_axis_num):
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0, 0, 0, 0, 0, 0)
#
#     axes_assembly.set_prop_values('ambient', 1.0)
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.5, axis=0 if use_axis_num else 'x')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val.x_shaft == 0.5
#     assert val.x_tip == 0.5
#     assert val == (0.5, 1.0, 1.0, 0.5, 1.0, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.7, axis=1 if use_axis_num else 'y', part='tip')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0.5, 1.0, 1.0, 0.5, 0.7, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.1, axis=2 if use_axis_num else 'z', part='shaft')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0.5, 1.0, 0.1, 0.5, 0.7, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.1, axis=2 if use_axis_num else 'z', part='shaft')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0.5, 1.0, 0.1, 0.5, 0.7, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.0, part='shaft')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0.0, 0.0, 0.0, 0.5, 0.7, 1.0)
#
#     axes_assembly.set_prop_values('ambient', 0.0, part='tip')
#     val = axes_assembly.get_prop_values('ambient')
#     assert val == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#
#     msg = "Part must be one of [0, 1, 'shaft', 'tip', 'all']."
#     with pytest.raises(ValueError, match=re.escape(msg)):
#         axes_assembly.set_prop_values('ambient', 0.0, part=2)
#
#     msg = "Axis must be one of [0, 1, 2, 'x', 'y', 'z', 'all']."
#     with pytest.raises(ValueError, match=re.escape(msg)):
#         axes_assembly.set_prop_values('ambient', 0.0, axis='a')


# def test_axes_assemblys(axes_assembly):
#     # test prop objects are distinct
#     props = axes_assembly._props
#     for i in range(6):
#         for j in range(6):
#             if i == j:
#                 assert props[i] is props[j]
#             else:
#                 assert props[i] is not props[j]
#
#     # test setting new prop
#     new_prop = pv.Property()
#     axes_assembly.x_shaft_prop = new_prop
#     assert axes_assembly.x_shaft_prop is new_prop
#     axes_assembly.y_shaft_prop = new_prop
#     assert axes_assembly.y_shaft_prop is new_prop
#     axes_assembly.z_shaft_prop = new_prop
#     assert axes_assembly.z_shaft_prop is new_prop
#     axes_assembly.x_tip_prop = new_prop
#     assert axes_assembly.x_tip_prop is new_prop
#     axes_assembly.y_tip_prop = new_prop
#     assert axes_assembly.y_tip_prop is new_prop
#     axes_assembly.z_tip_prop = new_prop
#     assert axes_assembly.z_tip_prop is new_prop
#
#     msg = "Prop must have type <class 'pyvista.plotting._property.Property'>, got <class 'int'> instead."
#     with pytest.raises(TypeError, match=msg):
#         axes_assembly.x_shaft_prop = 0


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
def test_axes_geometry_source_theme(axes_assembly):
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


# @pytest.mark.parametrize('tip_type', ['sphere', 'cone'])
# @pytest.mark.parametrize('radius', [(0.01, 0.4), (0.01, 0.4)])  # ,(1, 0),(0, 1)])
# @pytest.mark.parametrize('shaft_length', [[0.8, 0.8, 0.8]])
# @pytest.mark.parametrize('total_length', [[1, 1, 1], [4, 3, 2], [0.4, 0.5, 1.1]])
# @pytest.mark.parametrize('scale', [1, 2, 0.5])  # , [0.1, 0.2, 0.3], [2, 3, 4]])
# @pytest.mark.parametrize('auto_length', [True, False])  # ,False])
# def test_axes_assembly_true_to_scale(tip_type, radius, shaft_length, total_length, scale, auto_length):
#     shaft_radius, tip_radius = radius
#
#     kwargs = dict(
#         shaft_type='cylinder',
#         tip_type=tip_type,
#         shaft_length=shaft_length,
#         total_length=total_length,
#         scale=scale,
#         shaft_radius=shaft_radius,
#         tip_radius=tip_radius,
#         auto_length=auto_length,
#     )
#     # # Create reference axes actor
#     # axes_assembly = AxesActor(
#     #     true_to_scale=False,
#     #     **kwargs
#     # )
#     # bounds = [actor.GetBounds() for actor in axes_assembly._actors]
#     # normal_size = np.array([[b[1]-b[0],b[3]-b[2], b[5]-b[4]] for b in bounds])
#     # shaft_scale = normal_size[:3].copy()
#     # shaft_scale[np.invert(np.eye(3,dtype=bool))] /= (shaft_radius * 2)
#     #
#     # tip_scale = normal_size[3:6].copy()
#     # tip_scale[np.invert(np.eye(3,dtype=bool))] /= (tip_radius * 2)
#
#     # Create test
#     axes_assembly = AxesActor(true_to_scale=True, **kwargs)
#     # bounds = [actor.GetBounds() for actor in axes_assembly_true._actors]
#     # true_size = np.array([[b[1]-b[0],b[3]-b[2], b[5]-b[4]] for b in bounds])
#
#     #
#     shaft_length_scaled = np.array(shaft_length) * np.array(total_length) * np.array(scale)
#     tip_length_scaled = np.array(axes_assembly.tip_length) * np.array(total_length) * np.array(scale)
#
#     # actual_scale = axes_assembly._compute_true_to_scale_factors()
#     # assert actual_scale.x_shaft * shaft_length_scaled[0] == 1.0
#     # assert actual_scale.y_shaft * shaft_length_scaled[1] == 1.0
#     # assert actual_scale.z_shaft * shaft_length_scaled[2] == 1.0
#     # assert actual_scale.x_tip * tip_length_scaled[0] == 1.0
#     # assert actual_scale.y_tip * tip_length_scaled[1] == 1.0
#     # assert actual_scale.z_tip * tip_length_scaled[2] == 1.0
#
#     # axes_assembly._update_props()
#     xsl, ysl, zsl = shaft_length_scaled
#     xtl, ytl, ztl = tip_length_scaled
#     sr, tr = shaft_radius * scale, tip_radius * scale
#
#     # test x shaft
#     expected_bounds = (0.0, xsl, -sr, sr, -sr, sr)
#     actual_bounds = axes_assembly._actors.x_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test y shaft
#     expected_bounds = (-sr, sr, 0, ysl, -sr, sr)
#     actual_bounds = axes_assembly._actors.y_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test z shaft
#     expected_bounds = (-sr, sr, -sr, sr, 0, zsl)
#     actual_bounds = axes_assembly._actors.z_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test x tip
#     expected_bounds = (xsl, xsl + xtl, -tr, tr, -tr, tr)
#     actual_bounds = axes_assembly._actors.x_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)
#
#     # test y tip
#     expected_bounds = (-tr, tr, ysl, ysl + ytl, -tr, tr)
#     actual_bounds = axes_assembly._actors.y_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)
#
#     # test z tip
#     expected_bounds = (-tr, tr, -tr, tr, zsl, zsl + ztl)
#     actual_bounds = axes_assembly._actors.z_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)


def test_axes_assembly_output():
    out = pv.AxesAssembly()
    assert isinstance(out, pv.AxesAssembly)
