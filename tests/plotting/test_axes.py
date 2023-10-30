import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import array_from_vtkmatrix, vtkmatrix_from_array
from pyvista.plotting.axes import Axes
from pyvista.plotting.axes_actor import AxesActor
from pyvista.plotting.tools import create_axes_marker


@pytest.fixture(autouse=True)
def skip_check_gc(skip_check_gc):
    """All the tests here fail gc."""
    pass


@pytest.fixture()
def axes():
    return Axes()


@pytest.fixture()
def axes_actor():
    return AxesActor()


@pytest.fixture()
def vtk_axes_actor():
    return vtk.vtkAxesActor()


def test_actor_visibility(axes):
    # test showing
    assert not axes.actor.visibility
    axes.show_actor()
    assert axes.actor.visibility

    # test hiding
    assert axes.actor.visibility
    axes.hide_actor()
    assert not axes.actor.visibility


def test_axes_origin(axes):
    origin = np.random.random(3)
    axes.origin = origin
    assert np.all(axes.GetOrigin() == origin)
    assert np.all(axes.origin == origin)


def test_axes_symmetric(axes):
    # test showing
    assert not axes.GetSymmetric()
    axes.show_symmetric()
    assert axes.GetSymmetric()

    # test hiding
    assert axes.GetSymmetric()
    axes.hide_symmetric()
    assert not axes.GetSymmetric()


def test_axes_actor_visibility(axes_actor):
    assert axes_actor.visibility
    axes_actor.visibility = False
    assert not axes_actor.visibility

    actor_init = AxesActor(visibility=False)
    assert actor_init.visibility is False


def test_axes_actor_total_length(axes_actor):
    axes_actor.total_length = 2
    assert axes_actor.total_length == (2, 2, 2)

    axes_actor.total_length = (1, 2, 3)
    assert axes_actor.total_length == (1, 2, 3)

    actor_init = AxesActor(total_length=9)
    assert actor_init.total_length == (9, 9, 9)


def test_axes_actor_shaft_length(axes_actor):
    axes_actor.shaft_length = 1
    assert axes_actor.shaft_length == (1, 1, 1)

    axes_actor.shaft_length = (1, 2, 3)
    assert axes_actor.shaft_length == (1, 2, 3)

    actor_init = AxesActor(shaft_length=9)
    assert actor_init.shaft_length == (9, 9, 9)


def test_axes_actor_tip_length(axes_actor):
    axes_actor.tip_length = 1
    assert axes_actor.tip_length == (1, 1, 1)

    axes_actor.tip_length = (1, 2, 3)
    assert axes_actor.tip_length == (1, 2, 3)

    actor_init = AxesActor(tip_length=9)
    assert actor_init.tip_length == (9, 9, 9)


def test_axes_actor_label_position(axes_actor):
    axes_actor.label_position = 1
    assert axes_actor.label_position == (1, 1, 1)

    axes_actor.label_position = (1, 2, 3)
    assert axes_actor.label_position == (1, 2, 3)

    actor_init = AxesActor(label_position=9)
    assert actor_init.label_position == (9, 9, 9)


def test_axes_actor_tip_resolution(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove this deprecated property')

    axes_actor.tip_resolution = 42
    assert axes_actor.tip_resolution == 42

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_resolution` instead."):
        axes_actor.cone_resolution = 24
        assert axes_actor.cone_resolution == 24

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_resolution` instead."):
        axes_actor.sphere_resolution = 24
        assert axes_actor.sphere_resolution == 24

    actor_init = AxesActor(tip_resolution=42)
    assert actor_init.tip_resolution == 42


def test_axes_actor_shaft_resolution(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove this deprecated property')

    axes_actor.shaft_resolution = 42
    assert axes_actor.shaft_resolution == 42

    with pytest.warns(PyVistaDeprecationWarning, match="Use `shaft_resolution` instead."):
        axes_actor.cylinder_resolution = 24
        assert axes_actor.cylinder_resolution == 24

    actor_init = AxesActor(shaft_resolution=42)
    assert actor_init.shaft_resolution == 42


def test_axes_actor_tip_radius(axes_actor):
    axes_actor.tip_radius = 0.8
    assert axes_actor.tip_radius == 0.8
    assert axes_actor.GetConeRadius() == 0.8
    assert axes_actor.GetSphereRadius() == 0.8

    actor_init = AxesActor(tip_radius=9)
    assert actor_init.tip_radius == 9


def test_axes_actor_cone_radius(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove this deprecated property')

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_radius` instead."):
        axes_actor.cone_radius = 0.8
        assert axes_actor.cone_radius == 0.8


def test_axes_actor_sphere_radius(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove this deprecated property')

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_radius` instead."):
        axes_actor.sphere_radius = 0.8
        assert axes_actor.sphere_radius == 0.8


def test_axes_actor_cylinder_radius(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove this deprecated property')

    with pytest.warns(PyVistaDeprecationWarning, match="Use `shaft_radius` instead."):
        axes_actor.cylinder_radius = 0.03
        assert axes_actor.cylinder_radius == 0.03


def test_axes_actor_shaft_type(axes_actor):
    axes_actor.shaft_type = pv.AxesActor.ShaftType.CYLINDER
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.CYLINDER
    axes_actor.shaft_type = pv.AxesActor.ShaftType.LINE
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.LINE

    axes_actor.shaft_type = "cylinder"
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.CYLINDER
    axes_actor.shaft_type = "line"
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.LINE

    actor_init = AxesActor(shaft_type="cylinder")
    assert actor_init.shaft_type.annotation == "cylinder"
    actor_init = AxesActor(shaft_type="line")
    assert actor_init.shaft_type.annotation == "line"


def test_axes_actor_tip_type(axes_actor):
    axes_actor.tip_type = pv.AxesActor.TipType.CONE
    assert axes_actor.tip_type == pv.AxesActor.TipType.CONE
    axes_actor.tip_type = pv.AxesActor.TipType.SPHERE
    assert axes_actor.tip_type == pv.AxesActor.TipType.SPHERE

    axes_actor.tip_type = "cone"
    assert axes_actor.tip_type == pv.AxesActor.TipType.CONE
    axes_actor.tip_type = "sphere"
    assert axes_actor.tip_type == pv.AxesActor.TipType.SPHERE

    actor_init = AxesActor(tip_type="cone")
    assert actor_init.tip_type.annotation == "cone"
    actor_init = AxesActor(tip_type="sphere")
    assert actor_init.tip_type.annotation == "sphere"


def test_axes_actor_axes_labels(axes_actor):
    axes_actor.x_label = 'A'
    assert axes_actor.x_label == 'A'
    axes_actor.y_label = 'B'
    assert axes_actor.y_label == 'B'
    axes_actor.z_label = 'C'
    assert axes_actor.z_label == 'C'

    actor_init = AxesActor(x_label='A', y_label='B', z_label='C')
    assert actor_init.x_label == 'A'
    assert actor_init.y_label == 'B'
    assert actor_init.z_label == 'C'

    actor_init = AxesActor(xlabel='A', ylabel='B', z_label='C')
    assert actor_init.x_label == 'A'
    assert actor_init.y_label == 'B'
    assert actor_init.z_label == 'C'

    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove "xyz_axis_label" deprecated property')

    with pytest.warns(PyVistaDeprecationWarning, match="Use `x_label` instead."):
        axes_actor.x_axis_label = 'Axis X'
        assert axes_actor.x_axis_label == 'Axis X'
    with pytest.warns(PyVistaDeprecationWarning, match="Use `y_label` instead."):
        axes_actor.y_axis_label = 'Axis Y'
        assert axes_actor.y_axis_label == 'Axis Y'
    with pytest.warns(PyVistaDeprecationWarning, match="Use `z_label` instead."):
        axes_actor.z_axis_label = 'Axis Z'
        assert axes_actor.z_axis_label == 'Axis Z'


def test_axes_actor_label_color(axes_actor):
    axes_actor.label_color = 'purple'
    assert axes_actor.label_color.name == 'purple'
    axes_actor.label_color = [1, 2, 3]
    assert np.array_equal(axes_actor.label_color.int_rgb, [1, 2, 3])

    actor_init = AxesActor(label_color='yellow')
    assert actor_init.label_color.name == 'yellow'


def test_axes_actor_axis_color(axes_actor):
    axes_actor.x_color = 'purple'
    assert axes_actor.x_color.name == 'purple'
    axes_actor.x_color = [1, 2, 3]
    assert np.array_equal(axes_actor.x_color.int_rgb, [1, 2, 3])

    axes_actor.y_color = 'purple'
    assert axes_actor.y_color.name == 'purple'
    axes_actor.y_color = [1, 2, 3]
    assert np.array_equal(axes_actor.y_color.int_rgb, [1, 2, 3])

    axes_actor.z_color = 'purple'
    assert axes_actor.z_color.name == 'purple'
    axes_actor.z_color = [1, 2, 3]
    assert np.array_equal(axes_actor.z_color.int_rgb, [1, 2, 3])

    actor_init = AxesActor(x_color='yellow', y_color='orange', z_color='purple')
    assert actor_init.x_color.name == 'yellow'
    assert actor_init.y_color.name == 'orange'
    assert actor_init.z_color.name == 'purple'


def test_axes_shaft_width(axes_actor):
    axes_actor.shaft_width = 100

    assert axes_actor.shaft_width == 100
    assert axes_actor.GetXAxisShaftProperty().GetLineWidth() == 100
    assert axes_actor.GetYAxisShaftProperty().GetLineWidth() == 100
    assert axes_actor.GetZAxisShaftProperty().GetLineWidth() == 100

    actor_init = AxesActor(shaft_width=50)
    assert actor_init.shaft_width == 50


def test_axes_shaft_radius(axes_actor):
    axes_actor.shaft_radius = 100
    assert axes_actor.shaft_radius == 100

    actor_init = AxesActor(shaft_radius=50)
    assert actor_init.shaft_radius == 50


def test_axes_labels_off(axes_actor):
    axes_actor.labels_off = False
    assert axes_actor.labels_off is False
    axes_actor.labels_off = True
    assert axes_actor.labels_off is True

    actor_init = AxesActor(labels_off=True)
    assert actor_init.labels_off is True


def test_axes_label_size(axes_actor):
    w, h = 1, 2
    axes_actor.label_size = (w, h)
    assert axes_actor.label_size == (w, h)
    assert axes_actor.GetXAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetXAxisCaptionActor2D().GetHeight() == h
    assert axes_actor.GetYAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetYAxisCaptionActor2D().GetHeight() == h
    assert axes_actor.GetZAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetZAxisCaptionActor2D().GetHeight() == h

    actor_init = AxesActor(label_size=(2, 3))
    assert actor_init.label_size == (2, 3)


def test_axes_actor_properties(axes_actor):
    x_shaft = axes_actor.x_axis_shaft_properties
    x_shaft.ambient = 0.1
    assert axes_actor.x_axis_shaft_properties.ambient == 0.1
    x_tip = axes_actor.x_axis_tip_properties
    x_tip.ambient = 0.11
    assert axes_actor.x_axis_tip_properties.ambient == 0.11

    y_shaft = axes_actor.y_axis_shaft_properties
    y_shaft.ambient = 0.2
    assert axes_actor.y_axis_shaft_properties.ambient == 0.2
    y_tip = axes_actor.y_axis_tip_properties
    y_tip.ambient = 0.12
    assert axes_actor.y_axis_tip_properties.ambient == 0.12

    z_shaft = axes_actor.z_axis_shaft_properties
    z_shaft.ambient = 0.3
    assert axes_actor.z_axis_shaft_properties.ambient == 0.3
    z_tip = axes_actor.z_axis_tip_properties
    z_tip.ambient = 0.13
    assert axes_actor.z_axis_tip_properties.ambient == 0.13


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


@pytest.mark.parametrize('case', range(len(get_matrix_cases())))
def test_axes_actor_enable_orientation(axes_actor, vtk_axes_actor, case):
    # NOTE: This test works by asserting that:
    #   all(vtkAxesActor.GetMatrix()) == all(axes_actor.GetUserMatrix())
    #
    # Normally, GetUserMatrix() and GetMatrix() are very different matrices:
    # - GetUserMatrix() is an independent user-provided transformation matrix
    # - GetMatrix() concatenates
    #     implicit_transform -> user_transform -> coordinate system transform
    # However, as a workaround for pyvista#5019, UserMatrix is used
    # to represent the user_transform *and* implicit_transform.
    # Since the renderer's coordinate system transform is identity
    # by default, this means that in for this test the assertion should
    # hold true.

    cases = get_matrix_cases()
    angle = 42
    origin = (1, 5, 10)
    scale = (4, 5, 6)
    orientation = (10, 20, 30)
    position = (7, 8, 9)
    user_matrix = vtkmatrix_from_array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [0, 0, 0, 1]]
    )

    # test each property separately and also all together
    axes_actor._enable_orientation_workaround = True
    if case in [cases.ALL, cases.ORIENTATION]:
        vtk_axes_actor.SetOrientation(*orientation)
        axes_actor.orientation = orientation
    if case in [cases.ALL, cases.ORIGIN]:
        vtk_axes_actor.SetOrigin(*origin)
        axes_actor.origin = origin
    if case in [cases.ALL, cases.SCALE]:
        vtk_axes_actor.SetScale(*scale)
        axes_actor.scale = scale
    if case in [cases.ALL, cases.POSITION]:
        vtk_axes_actor.SetPosition(*position)
        axes_actor.position = position
    if case in [cases.ALL, cases.ROTATE_X]:
        vtk_axes_actor.RotateX(angle)
        axes_actor.rotate_x(angle)
    if case in [cases.ALL, cases.ROTATE_Y]:
        vtk_axes_actor.RotateY(angle * 2)
        axes_actor.rotate_y(angle * 2)
    if case in [cases.ALL, cases.ROTATE_Z]:
        vtk_axes_actor.RotateZ(angle * 3)
        axes_actor.rotate_z(angle * 3)
    if case in [cases.ALL, cases.USER_MATRIX]:
        vtk_axes_actor.SetUserMatrix(user_matrix)
        axes_actor.user_matrix = user_matrix

    expected = array_from_vtkmatrix(vtk_axes_actor.GetMatrix())
    actual = array_from_vtkmatrix(axes_actor.GetUserMatrix())
    assert np.allclose(expected, actual)

    default_bounds = (-1, 1, -1, 1, -1, 1)
    assert np.allclose(vtk_axes_actor.GetBounds(), default_bounds)

    # test AxesActor has non-default bounds except in ORIGIN case
    actual_bounds = axes_actor.GetBounds()
    if case == cases.ORIGIN:
        assert np.allclose(actual_bounds, default_bounds)
    else:
        assert not np.allclose(actual_bounds, default_bounds)

    # test that bounds are always default (i.e. incorrect) after disabling orientation
    axes_actor._enable_orientation_workaround = False
    actual_bounds = axes_actor.GetBounds()
    assert np.allclose(actual_bounds, default_bounds)


def test_axes_actor_deprecated_constructor():
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Remove deprecated AxesActor constructor `create_axes_marker`')

    # test create_axes_marker raises deprecation error
    with pytest.warns(PyVistaDeprecationWarning, match='`create_axes_actor` has been deprecated'):
        create_axes_marker()

    # test that deprecated args used by `create_axes_marker` are handled correctly by AxesActor
    # define valid default args originally used by `create_axes_marker` in PyVista version < 0.43.0
    old_args = dict(
        label_color='yellow',
        x_color='red',
        y_color='green',
        z_color='blue',
        xlabel='A',
        ylabel='B',
        zlabel='C',
        labels_off=False,
        line_width=20,
        cone_radius=0.9,
        shaft_length=0.9,
        tip_length=0.1,
        ambient=0.9,
        label_size=(0.25, 0.1),
    )
    old_args_modified = old_args.copy()
    with pytest.warns(
        PyVistaDeprecationWarning,
        match="Use `tip_radius` instead.",
    ):
        axes_actor = AxesActor(**old_args_modified)
    old_args_modified["tip_radius"] = old_args_modified.pop("cone_radius")

    with pytest.warns(
        PyVistaDeprecationWarning,
        match=f"Use `properties={{'ambient':{old_args_modified['ambient']}}}` instead.",
    ):
        axes_actor = AxesActor(**old_args_modified)
    old_args_modified["properties"] = {"ambient": old_args_modified.pop("ambient")}

    with pytest.warns(
        PyVistaDeprecationWarning,
        match="Use `shaft_width` instead.",
    ):
        axes_actor = AxesActor(**old_args_modified)
    old_args_modified["shaft_width"] = old_args_modified.pop("line_width")

    axes_actor = AxesActor(**old_args_modified)

    assert old_args["label_color"] == axes_actor.label_color.name
    assert old_args["x_color"] == axes_actor.x_color.name
    assert old_args["y_color"] == axes_actor.y_color.name
    assert old_args["z_color"] == axes_actor.z_color.name
    assert old_args["xlabel"] == axes_actor.x_label
    assert old_args["ylabel"] == axes_actor.y_label
    assert old_args["zlabel"] == axes_actor.z_label
    assert old_args["labels_off"] == axes_actor.labels_off
    assert old_args["line_width"] == axes_actor.shaft_width
    assert old_args["cone_radius"] == axes_actor.tip_radius
    assert tuple([old_args["shaft_length"]] * 3) == axes_actor.shaft_length
    assert tuple([old_args["tip_length"]] * 3) == axes_actor.tip_length
    assert old_args["ambient"] == axes_actor.x_axis_shaft_properties.ambient
    assert old_args["label_size"] == axes_actor.label_size


def test_axes_actor_raises():
    with pytest.raises(TypeError, match="must be a dictionary"):
        pv.AxesActor(properties="not_a_dict")
    with pytest.raises(TypeError, match="invalid keyword"):
        pv.AxesActor(not_valid_kwarg="some_value")
