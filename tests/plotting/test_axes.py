from __future__ import annotations

import re

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.plotting.axes import Axes
from pyvista.plotting.axes_actor import AxesActor
from pyvista.plotting.axes_actor import AxesActorComposite
from pyvista.plotting.tools import create_axes_marker


@pytest.fixture(autouse=True)
def skip_check_gc(skip_check_gc):  # noqa: PT004
    """All the tests here fail gc."""


@pytest.fixture()
def axes():
    return Axes()


@pytest.fixture()
def axes_actor():
    return AxesActor()


@pytest.fixture()
def vtk_axes_actor():
    return vtk.vtkAxesActor()


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'AxesActor_from_base' in metafunc.fixturenames:
        classes = [AxesActor, AxesActorComposite]
        ids = [clas.__name__ for clas in classes]

        # Make sure classes can be initialized
        assert AxesActor()
        assert AxesActorComposite()

        metafunc.parametrize('AxesActor_from_base', classes, ids=ids)


def test_actor_visibility(axes):
    # test showing
    assert not axes.actor.visibility
    axes.show_actor()
    assert axes.actor.visibility

    # test hiding
    assert axes.actor.visibility
    axes.hide_actor()
    assert not axes.actor.visibility


def test_origin(axes):
    origin = np.random.default_rng().random(3)
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


def test_axes_actor_base_visibility(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.visibility

    axes_actor = AxesActor_from_base(visibility=False)
    assert not axes_actor.visibility

    axes_actor.visibility = True
    assert axes_actor.visibility


def test_axes_actor_base_total_length(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.total_length == (1, 1, 1)

    axes_actor = AxesActor_from_base(total_length=2)
    assert axes_actor.total_length == (2, 2, 2)

    axes_actor.total_length = (1, 2, 3)
    assert axes_actor.total_length == (1, 2, 3)


def test_axes_actor_base_shaft_length(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.shaft_length == (0.8, 0.8, 0.8)

    axes_actor = AxesActor_from_base(shaft_length=0.9)
    assert axes_actor.shaft_length == (0.9, 0.9, 0.9)

    axes_actor.shaft_length = (0.1, 0.2, 0.3)
    assert axes_actor.shaft_length == (0.1, 0.2, 0.3)


def test_axes_actor_base_tip_length(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.tip_length == (0.2, 0.2, 0.2)

    axes_actor = AxesActor_from_base(tip_length=0.9)
    assert axes_actor.tip_length == (0.9, 0.9, 0.9)

    axes_actor.tip_length = (0.1, 0.2, 0.3)
    assert axes_actor.tip_length == (0.1, 0.2, 0.3)


def test_axes_actor_base_label_position(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.label_position == (1, 1, 1)

    axes_actor = AxesActor_from_base(label_position=2)
    assert axes_actor.label_position == (2, 2, 2)

    axes_actor.label_position = (1, 2, 3)
    assert axes_actor.label_position == (1, 2, 3)


def test_axes_actor_base_tip_resolution(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.tip_resolution == 24

    actor_init = AxesActor_from_base(tip_resolution=42)
    assert actor_init.tip_resolution == 42

    axes_actor.tip_resolution = 99
    assert axes_actor.tip_resolution == 99


def test_axes_actor_deprecated_parameters(axes_actor):
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Convert this deprecation warning to an error.')
    if pv._version.version_info >= (0, 47):
        raise RuntimeError(
            'Remove deprecated properties:\n'
            'cone_resolution\n'
            'sphere_resolution\n'
            'shaft_resolution\n'
            'cone_radius\n'
            'sphere_radius\n'
            'cylinder_radius\n'
            'x_axis_label\n'
            'y_axis_label\n'
            'z_axis_label\n'
            'x_axis_shaft_properties\n'
            'y_axis_shaft_properties\n'
            'z_axis_shaft_properties\n'
            'x_axis_tip_properties\n'
            'y_axis_tip_properties\n'
            'z_axis_tip_properties\n',
        )

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_resolution` instead."):
        axes_actor.cone_resolution = 24
        assert axes_actor.cone_resolution == 24

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_resolution` instead."):
        axes_actor.sphere_resolution = 24
        assert axes_actor.sphere_resolution == 24

    with pytest.warns(PyVistaDeprecationWarning, match="Use `shaft_resolution` instead."):
        axes_actor.cylinder_resolution = 24
        assert axes_actor.cylinder_resolution == 24

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_radius` instead."):
        axes_actor.cone_radius = 0.8
        assert axes_actor.cone_radius == 0.8

    with pytest.warns(PyVistaDeprecationWarning, match="Use `tip_radius` instead."):
        axes_actor.sphere_radius = 0.8
        assert axes_actor.sphere_radius == 0.8

    with pytest.warns(PyVistaDeprecationWarning, match="Use `shaft_radius` instead."):
        axes_actor.cylinder_radius = 0.03
        assert axes_actor.cylinder_radius == 0.03

    with pytest.warns(PyVistaDeprecationWarning, match="Use `x_label` instead."):
        axes_actor.x_axis_label = 'Axis X'
        assert axes_actor.x_axis_label == 'Axis X'

    with pytest.warns(PyVistaDeprecationWarning, match="Use `y_label` instead."):
        axes_actor.y_axis_label = 'Axis Y'
        assert axes_actor.y_axis_label == 'Axis Y'

    with pytest.warns(PyVistaDeprecationWarning, match="Use `z_label` instead."):
        axes_actor.z_axis_label = 'Axis Z'
        assert axes_actor.z_axis_label == 'Axis Z'

    with pytest.warns(PyVistaDeprecationWarning, match="Use `x_shaft_prop` instead."):
        _ = axes_actor.x_axis_shaft_properties

    with pytest.warns(PyVistaDeprecationWarning, match="Use `y_shaft_prop` instead."):
        _ = axes_actor.y_axis_shaft_properties

    with pytest.warns(PyVistaDeprecationWarning, match="Use `z_shaft_prop` instead."):
        _ = axes_actor.z_axis_shaft_properties

    with pytest.warns(PyVistaDeprecationWarning, match="Use `x_tip_prop` instead."):
        _ = axes_actor.x_axis_tip_properties

    with pytest.warns(PyVistaDeprecationWarning, match="Use `y_tip_prop` instead."):
        _ = axes_actor.y_axis_tip_properties

    with pytest.warns(PyVistaDeprecationWarning, match="Use `z_tip_prop` instead."):
        _ = axes_actor.z_axis_tip_properties


def test_axes_actor_deprecated_enums(axes_actor):
    pass
    # TODO: test deprecation
    # axes_actor.shaft_type = pv.AxesActor.ShaftType.CYLINDER
    # assert axes_actor.shaft_type == pv.AxesActor.ShaftType.CYLINDER
    # axes_actor.shaft_type = pv.AxesActor.ShaftType.LINE
    # assert axes_actor.shaft_type == pv.AxesActor.ShaftType.LINE

    # axes_actor.tip_type = pv.AxesActor.TipType.CONE
    # assert axes_actor.tip_type == pv.AxesActor.TipType.CONE
    # axes_actor.tip_type = pv.AxesActor.TipType.SPHERE
    # assert axes_actor.tip_type == pv.AxesActor.TipType.SPHERE


def test_axes_actor_base_shaft_resolution(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.shaft_resolution == 24

    actor_init = AxesActor_from_base(shaft_resolution=42)
    assert actor_init.shaft_resolution == 42

    axes_actor.shaft_resolution = 99
    assert axes_actor.shaft_resolution == 99


def test_axes_actor_base_tip_radius(AxesActor_from_base):
    axes_actor = AxesActor_from_base(tip_radius=9)
    assert axes_actor.tip_radius == 9

    axes_actor.tip_radius = 0.8
    assert axes_actor.tip_radius == 0.8


def test_axes_actor_tip_radius():
    axes_actor = AxesActor()
    axes_actor.tip_radius = 0.8
    assert axes_actor.tip_radius == 0.8
    assert axes_actor.GetConeRadius() == 0.8
    assert axes_actor.GetSphereRadius() == 0.8

    # test that bounds are correct when radius >> length
    axes_actor.tip_radius = 80
    assert np.allclose(axes_actor.bounds, (-16, 16, -16, 16, -16, 16))

    # test that changing radius correctly updates bounds
    axes_actor.tip_radius = 30
    assert np.allclose(axes_actor.bounds, (-6, 6, -6, 6, -6, 6))


def test_axes_actor_base_shaft_type(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.shaft_type == "cylinder"

    axes_actor = AxesActor_from_base(shaft_type="cylinder")
    assert axes_actor.shaft_type == "cylinder"


def test_axes_actor_shaft_type():
    axes_actor = AxesActor(shaft_type="line")
    assert axes_actor.shaft_type == "line"

    axes_actor.shaft_type = 'cylinder'
    assert axes_actor.shaft_type == 'cylinder'
    axes_actor.shaft_type = 'line'
    assert axes_actor.shaft_type == 'line'


def test_axes_actor_tip_type(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.tip_type == 'cone'

    axes_actor = AxesActor_from_base(tip_type='cone')
    assert axes_actor.tip_type == 'cone'

    axes_actor.tip_type = "sphere"
    assert axes_actor.tip_type == 'sphere'
    axes_actor.tip_type = "cone"
    assert axes_actor.tip_type == 'cone'


def test_axes_actor_labels(AxesActor_from_base):
    axes_actor = AxesActor_from_base()
    assert axes_actor.x_label == 'X'
    assert axes_actor.y_label == 'Y'
    assert axes_actor.z_label == 'Z'

    axes_actor = AxesActor_from_base(x_label='A', y_label='B', z_label='C')
    assert axes_actor.x_label == 'A'
    assert axes_actor.y_label == 'B'
    assert axes_actor.z_label == 'C'

    axes_actor = AxesActor_from_base(x_label='A', y_label='B', z_label='C', labels='UVW')
    assert axes_actor.x_label == 'U'
    assert axes_actor.y_label == 'V'
    assert axes_actor.z_label == 'W'

    axes_actor.x_label = '1'
    assert axes_actor.x_label == '1'
    axes_actor.y_label = '2'
    assert axes_actor.y_label == '2'
    axes_actor.z_label = '3'
    assert axes_actor.z_label == '3'

    axes_actor.labels = ('U', 'V', 'W')
    assert axes_actor.x_label == 'U'
    assert axes_actor.y_label == 'V'
    assert axes_actor.z_label == 'W'

    axes_actor.labels = 'UVW'
    assert axes_actor.labels == ('U', 'V', 'W')
    assert axes_actor.x_label == 'U'
    assert axes_actor.y_label == 'V'
    assert axes_actor.z_label == 'W'

    with pytest.raises(ValueError, match='Labels sequence must have exactly 3 items.'):
        axes_actor.labels = 'abcd'


def test_axes_actor_base_label_color(AxesActor_from_base):
    axes_actor = AxesActor_from_base(label_color='yellow')
    assert axes_actor.label_color[0].name == 'yellow'
    assert axes_actor.label_color[1].name == 'yellow'
    assert axes_actor.label_color[2].name == 'yellow'

    axes_actor.label_color = 'purple'
    assert len(axes_actor.label_color) == 3
    assert axes_actor.label_color[0].name == 'purple'
    assert axes_actor.label_color[1].name == 'purple'
    assert axes_actor.label_color[2].name == 'purple'

    axes_actor.label_color = 'rgb'
    assert len(axes_actor.label_color) == 3
    assert axes_actor.label_color[0].name == 'red'
    assert axes_actor.label_color[1].name == 'green'
    assert axes_actor.label_color[2].name == 'blue'

    axes_actor.label_color = ['red', 'green', 'blue']
    assert len(axes_actor.label_color) == 3
    assert axes_actor.label_color[0].name == 'red'
    assert axes_actor.label_color[1].name == 'green'
    assert axes_actor.label_color[2].name == 'blue'

    axes_actor.label_color = [1, 2, 3]
    assert len(axes_actor.label_color) == 3
    assert np.array_equal(axes_actor.label_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_actor.label_color[1].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_actor.label_color[2].int_rgb, [1, 2, 3])


def test_axes_actor_base_axis_color(AxesActor_from_base):
    axes_actor = AxesActor_from_base(x_color='yellow', y_color='orange', z_color='purple')
    assert axes_actor.x_color[0].name == 'yellow'
    assert axes_actor.x_color[1].name == 'yellow'
    assert axes_actor.y_color[0].name == 'orange'
    assert axes_actor.y_color[1].name == 'orange'
    assert axes_actor.z_color[0].name == 'purple'
    assert axes_actor.z_color[1].name == 'purple'

    axes_actor.x_color = 'purple'
    assert len(axes_actor.x_color) == 2
    assert axes_actor.x_color[0].name == 'purple'
    assert axes_actor.x_color[1].name == 'purple'

    axes_actor.x_color = [1, 2, 3]
    assert np.array_equal(axes_actor.x_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_actor.x_color[1].int_rgb, [1, 2, 3])

    axes_actor.y_color = 'purple'
    assert axes_actor.y_color[0].name == 'purple'
    assert axes_actor.y_color[1].name == 'purple'
    axes_actor.y_color = [1, 2, 3]
    assert np.array_equal(axes_actor.y_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_actor.y_color[1].int_rgb, [1, 2, 3])

    axes_actor.z_color = 'purple'
    assert axes_actor.z_color[0].name == 'purple'
    assert axes_actor.z_color[1].name == 'purple'
    axes_actor.z_color = [1, 2, 3]
    assert np.array_equal(axes_actor.z_color[0].int_rgb, [1, 2, 3])
    assert np.array_equal(axes_actor.z_color[1].int_rgb, [1, 2, 3])


def test_axes_actor_shaft_width():
    axes_actor = AxesActor(shaft_width=50)
    assert axes_actor.shaft_width == 50
    # test setting width automatically changes type to 'line'
    axes_actor.shaft_type = 'cylinder'
    axes_actor.shaft_width = 100

    assert axes_actor.shaft_type == 'line'
    assert axes_actor.shaft_width == 100
    assert axes_actor.GetXAxisShaftProperty().GetLineWidth() == 100
    assert axes_actor.GetYAxisShaftProperty().GetLineWidth() == 100
    assert axes_actor.GetZAxisShaftProperty().GetLineWidth() == 100


def test_axes_actor_base_shaft_radius(AxesActor_from_base):
    axes_actor = AxesActor_from_base(shaft_radius=3)
    assert axes_actor.shaft_radius == 3

    axes_actor.shaft_radius = 0.1
    assert axes_actor.shaft_radius == 0.1


def test_axes_actor_shaft_radius():
    axes_actor = AxesActor(shaft_radius=50)
    assert axes_actor.shaft_radius == 50

    # test setting width automatically changes type to 'cylinder'
    axes_actor.shaft_type = 'line'
    axes_actor.shaft_radius = 100

    assert axes_actor.shaft_type == 'cylinder'
    assert axes_actor.shaft_radius == 100

    # test that bounds are correct
    assert np.allclose(axes_actor.bounds, (-80, 80, -80, 80, -80, 80))

    # test that changing radius only correctly updates bounds
    axes_actor.shaft_radius = 0
    assert np.allclose(axes_actor.bounds, (-1, 1, -1, 1, -1, 1))

    axes_actor.shaft_radius = 30
    assert np.allclose(axes_actor.bounds, (-24, 24, -24, 24, -24, 24))


def test_axes_actor_labels_off(axes_actor):
    axes_actor.labels_off = False
    assert axes_actor.labels_off is False
    assert axes_actor.GetAxisLabels()

    axes_actor.labels_off = True
    assert axes_actor.labels_off is True
    assert not axes_actor.GetAxisLabels()

    actor_init = AxesActor(labels_off=True)
    assert actor_init.labels_off is True
    assert not axes_actor.GetAxisLabels()


def test_axes_actor_label_size(axes_actor):
    w, h = 0.1, 0.2
    axes_actor.label_size = (w, h)
    assert axes_actor.label_size == (w, h)
    assert axes_actor.GetXAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetXAxisCaptionActor2D().GetHeight() == h
    assert axes_actor.GetYAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetYAxisCaptionActor2D().GetHeight() == h
    assert axes_actor.GetZAxisCaptionActor2D().GetWidth() == w
    assert axes_actor.GetZAxisCaptionActor2D().GetHeight() == h

    actor_init = AxesActor(label_size=(0.2, 0.3))
    assert actor_init.label_size == (0.2, 0.3)


def test_axes_actor_properties(axes_actor):
    axes_actor.x_shaft_prop.ambient = 0.1
    assert axes_actor.x_shaft_prop.ambient == 0.1
    axes_actor.x_tip_prop.ambient = 0.11
    assert axes_actor.x_tip_prop.ambient == 0.11

    axes_actor.y_shaft_prop.ambient = 0.2
    assert axes_actor.y_shaft_prop.ambient == 0.2
    axes_actor.y_tip_prop.ambient = 0.12
    assert axes_actor.y_tip_prop.ambient == 0.12

    axes_actor.z_shaft_prop.ambient = 0.3
    assert axes_actor.z_shaft_prop.ambient == 0.3
    axes_actor.z_tip_prop.ambient = 0.13
    assert axes_actor.z_tip_prop.ambient == 0.13

    # Test init
    prop = pv.Property(ambient=0.42)
    axes_actor = AxesActor(properties=prop)
    assert axes_actor.x_shaft_prop.ambient == 0.42
    assert axes_actor.x_shaft_prop is not prop
    assert axes_actor.x_tip_prop.ambient == 0.42
    assert axes_actor.y_shaft_prop is not prop

    assert axes_actor.y_shaft_prop.ambient == 0.42
    assert axes_actor.y_shaft_prop is not prop
    assert axes_actor.y_tip_prop.ambient == 0.42
    assert axes_actor.y_shaft_prop is not prop

    assert axes_actor.z_shaft_prop.ambient == 0.42
    assert axes_actor.z_shaft_prop is not prop
    assert axes_actor.z_tip_prop.ambient == 0.42
    assert axes_actor.z_shaft_prop is not prop

    msg = '`properties` must be a property object or a dictionary.'
    with pytest.raises(TypeError, match=msg):
        pv.AxesActor(properties="not_a_dict")


def test_axes_actor_user_matrix():
    eye = np.eye(4)
    eye2 = eye * 2
    eye3 = eye * 3

    a = pv.AxesActor(_make_orientable=False)
    assert np.array_equal(a.user_matrix, eye)
    assert np.array_equal(a._user_matrix, eye)
    assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye)

    a.user_matrix = eye2
    assert np.array_equal(a.user_matrix, eye2)
    assert np.array_equal(a._user_matrix, eye2)
    assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye2)

    a._make_orientable = True
    a.SetUserMatrix(vtkmatrix_from_array(eye3))
    assert np.array_equal(a.user_matrix, eye3)
    assert np.array_equal(a._user_matrix, eye3)
    assert np.array_equal(array_from_vtkmatrix(a.GetUserMatrix()), eye3)


def _compute_expected_bounds(axes_actor):
    # Compute expected bounds by transforming vtkAxesActor actors

    # Create two vtkAxesActor (one for (+) and one for (-) axes)
    # and store their actors
    actors = []
    vtk_axes = [vtk.vtkAxesActor(), vtk.vtkAxesActor()]
    for axes in vtk_axes:
        axes.SetNormalizedShaftLength(axes_actor.shaft_length)
        axes.SetNormalizedTipLength(axes_actor.tip_length)
        axes.SetTotalLength(axes_actor.total_length)
        axes.SetShaftTypeToCylinder()

        props = vtk.vtkPropCollection()
        axes.GetActors(props)
        actors.extend([props.GetItemAsObject(num) for num in range(6)])

    # Transform actors and get their bounds
    # Half of the actors are reflected to give symmetric axes
    bounds = []
    matrix = axes_actor._concatenate_implicit_matrix_and_user_matrix()
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


@pytest.mark.parametrize('shaft_tip_length', [(0, 1), (0.2, 0.3), (0.3, 0.8), (0.4, 0.6), (1, 0)])
@pytest.mark.parametrize('total_length', [[1, 1, 1], [4, 3, 2], [0.4, 0.5, 1.1]])
@pytest.mark.parametrize('scale', [[1, 1, 1], [0.1, 0.2, 0.3], [2, 3, 4]])
@pytest.mark.parametrize('position', [[0, 0, 0], [2, 3, 4]])
def test_axes_actor_GetBounds(shaft_tip_length, total_length, scale, position):
    shaft_length, tip_length = shaft_tip_length

    # Test the override for GetBounds() returns the same result as without override
    # for zero-centered, axis-aligned cases (i.e. no position, scale, etc.)
    vtk_axes_actor = vtk.vtkAxesActor()
    vtk_axes_actor.SetNormalizedShaftLength(shaft_length, shaft_length, shaft_length)
    vtk_axes_actor.SetNormalizedTipLength(tip_length, tip_length, tip_length)
    vtk_axes_actor.SetTotalLength(total_length)
    vtk_axes_actor.SetShaftTypeToCylinder()
    expected = vtk_axes_actor.GetBounds()

    axes_actor = pv.AxesActor(
        shaft_length=shaft_length,
        tip_length=tip_length,
        total_length=total_length,
        auto_length=False,
        _make_orientable=True,
    )
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor._make_orientable = False
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    # test GetBounds() returns correct output when axes are orientable
    axes_actor._make_orientable = True
    axes_actor.position = position
    axes_actor.scale = scale
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    # test that changing properties dynamically updates the axes
    axes_actor.position = rng.random(3) * 2 - 1
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor.scale = rng.random(3) * 2
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor.user_matrix = np.diag(np.append(rng.random(3), 1))
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor.tip_length = rng.random(3)
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor.shaft_length = rng.random(3)
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    axes_actor.total_length = rng.random(3) * 2
    expected = _compute_expected_bounds(axes_actor)
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)

    # test disabling workaround correctly resets axes actors
    vtk_axes_actor = vtk.vtkAxesActor()
    vtk_axes_actor.SetPosition(axes_actor.position)
    vtk_axes_actor.SetScale(axes_actor.scale)
    vtk_axes_actor.SetNormalizedShaftLength(axes_actor.shaft_length)
    vtk_axes_actor.SetNormalizedTipLength(axes_actor.tip_length)
    vtk_axes_actor.SetTotalLength(axes_actor.total_length)
    vtk_axes_actor.SetShaftTypeToCylinder()

    axes_actor._make_orientable = False
    expected = vtk_axes_actor.GetBounds()
    actual = axes_actor.GetBounds()
    assert np.allclose(actual, expected)


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
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [0, 0, 0, 1]],
    )

    # test each property separately and also all together
    axes_actor._make_orientable = True
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
    actual = array_from_vtkmatrix(axes_actor._actors[0].GetUserMatrix())
    assert np.allclose(expected, actual)

    default_bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    assert np.allclose(vtk_axes_actor.GetBounds(), default_bounds)

    # test AxesActor has non-default bounds except in ORIGIN case
    actual_bounds = axes_actor.GetBounds()
    if case == cases.ORIGIN:
        assert np.allclose(actual_bounds, default_bounds)
    else:
        assert not np.allclose(actual_bounds, default_bounds)

    # test that bounds are always default (i.e. incorrect) after disabling orientation
    axes_actor._make_orientable = False
    actual_bounds = axes_actor.GetBounds()
    assert np.allclose(actual_bounds, default_bounds)


def test_axes_actor_deprecated_constructor():
    if pv._version.version_info >= (0, 46):
        raise RuntimeError('Convert this deprecation warning to an error.')
    if pv._version.version_info >= (0, 47):
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
        match="deprecated.",
    ):
        axes_actor = AxesActor(**old_args_modified)
    old_args_modified["tip_radius"] = old_args_modified.pop("cone_radius")

    with pytest.warns(
        PyVistaDeprecationWarning,
        match=re.escape("deprecated"),
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

    assert old_args["label_color"] == axes_actor.label_color[0].name
    assert old_args["x_color"] == axes_actor.x_color[0].name
    assert old_args["y_color"] == axes_actor.y_color[0].name
    assert old_args["z_color"] == axes_actor.z_color[0].name
    assert old_args["xlabel"] == axes_actor.x_label
    assert old_args["ylabel"] == axes_actor.y_label
    assert old_args["zlabel"] == axes_actor.z_label
    assert old_args["labels_off"] == axes_actor.labels_off
    assert old_args["line_width"] == axes_actor.shaft_width
    assert old_args["cone_radius"] == axes_actor.tip_radius
    assert tuple([old_args["shaft_length"]] * 3) == axes_actor.shaft_length
    assert tuple([old_args["tip_length"]] * 3) == axes_actor.tip_length
    assert old_args["ambient"] == axes_actor.x_shaft_prop.ambient
    assert old_args["label_size"] == axes_actor.label_size


def test_axes_actor_repr(axes_actor):
    repr_ = repr(axes_actor)
    actual_lines = repr_.splitlines()[1:]
    expected_lines = [
        "  X label:                    'X'",
        "  Y label:                    'Y'",
        "  Z label:                    'Z'",
        "  Labels off:                 False",
        "  Label position:             (1.0, 1.0, 1.0)",
        "  Shaft type:                 'cylinder'",
        "  Shaft radius:               0.015",
        "  Shaft length:               (0.8, 0.8, 0.8)",
        "  Tip type:                   'cone'",
        "  Tip radius:                 0.4",
        "  Tip length:                 (0.2, 0.2, 0.2)",
        "  Total length:               (1.0, 1.0, 1.0)",
        "  Position:                   (0.0, 0.0, 0.0)",
        "  Scale:                      (1.0, 1.0, 1.0)",
        "  User matrix:                Identity",
        "  Visible:                    True",
        "  X Bounds                    -1.000E+00, 1.000E+00",
        "  Y Bounds                    -1.000E+00, 1.000E+00",
        "  Z Bounds                    -1.000E+00, 1.000E+00",
    ]

    assert all(actual == expected for (actual, expected) in zip(actual_lines, expected_lines))
    assert len(actual_lines) == len(expected_lines)

    axes_actor.shaft_type = 'line'
    repr_ = repr(axes_actor)
    assert "'line'" in repr_

    axes_actor.user_matrix = np.eye(4) * 2
    repr_ = repr(axes_actor)
    assert "User matrix:                Set" in repr_


@pytest.mark.parametrize('use_axis_num', [True, False])
def test_axes_actor_set_get_prop(axes_actor, use_axis_num):
    val = axes_actor.get_prop('ambient')
    assert val == (0, 0, 0, 0, 0, 0)

    axes_actor.set_prop('ambient', 1.0)
    val = axes_actor.get_prop('ambient')
    assert val == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    axes_actor.set_prop('ambient', 0.5, axis=0 if use_axis_num else 'x')
    val = axes_actor.get_prop('ambient')
    assert val.x_shaft == 0.5
    assert val.x_tip == 0.5
    assert val == (0.5, 1.0, 1.0, 0.5, 1.0, 1.0)

    axes_actor.set_prop('ambient', 0.7, axis=1 if use_axis_num else 'y', part='tip')
    val = axes_actor.get_prop('ambient')
    assert val == (0.5, 1.0, 1.0, 0.5, 0.7, 1.0)

    axes_actor.set_prop('ambient', 0.1, axis=2 if use_axis_num else 'z', part='shaft')
    val = axes_actor.get_prop('ambient')
    assert val == (0.5, 1.0, 0.1, 0.5, 0.7, 1.0)

    axes_actor.set_prop('ambient', 0.1, axis=2 if use_axis_num else 'z', part='shaft')
    val = axes_actor.get_prop('ambient')
    assert val == (0.5, 1.0, 0.1, 0.5, 0.7, 1.0)

    axes_actor.set_prop('ambient', 0.0, part='shaft')
    val = axes_actor.get_prop('ambient')
    assert val == (0.0, 0.0, 0.0, 0.5, 0.7, 1.0)

    axes_actor.set_prop('ambient', 0.0, part='tip')
    val = axes_actor.get_prop('ambient')
    assert val == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    msg = "Part must be one of [0, 1, 'shaft', 'tip', 'all']."
    with pytest.raises(ValueError, match=re.escape(msg)):
        axes_actor.set_prop('ambient', 0.0, part=2)

    msg = "Axis must be one of [0, 1, 2, 'x', 'y', 'z', 'all']."
    with pytest.raises(ValueError, match=re.escape(msg)):
        axes_actor.set_prop('ambient', 0.0, axis='a')


def test_axes_actor_props(axes_actor):
    # test actors and props are stored correctly
    actors = vtk.vtkPropCollection()
    axes_actor.GetActors(actors)
    actors = [actors.GetItemAsObject(i) for i in range(6)]

    assert actors[0].GetProperty() is axes_actor._actor_properties[0]
    assert actors[1].GetProperty() is axes_actor._actor_properties[1]
    assert actors[2].GetProperty() is axes_actor._actor_properties[2]
    assert actors[3].GetProperty() is axes_actor._actor_properties[3]
    assert actors[4].GetProperty() is axes_actor._actor_properties[4]
    assert actors[5].GetProperty() is axes_actor._actor_properties[5]

    assert axes_actor.GetXAxisShaftProperty() is axes_actor.x_shaft_prop
    assert axes_actor.GetYAxisShaftProperty() is axes_actor.y_shaft_prop
    assert axes_actor.GetZAxisShaftProperty() is axes_actor.z_shaft_prop
    assert axes_actor.GetXAxisTipProperty() is axes_actor.x_tip_prop
    assert axes_actor.GetYAxisTipProperty() is axes_actor.y_tip_prop
    assert axes_actor.GetZAxisTipProperty() is axes_actor.z_tip_prop

    # test prop objects are distinct
    for i in range(6):
        for j in range(6):
            if i == j:
                assert axes_actor._actor_properties[i] is axes_actor._actor_properties[j]
            else:
                assert axes_actor._actor_properties[i] is not axes_actor._actor_properties[j]

    # test setting new prop
    new_prop = pv.Property()
    axes_actor.x_shaft_prop = new_prop
    assert axes_actor.x_shaft_prop is new_prop
    axes_actor.y_shaft_prop = new_prop
    assert axes_actor.y_shaft_prop is new_prop
    axes_actor.z_shaft_prop = new_prop
    assert axes_actor.z_shaft_prop is new_prop
    axes_actor.x_tip_prop = new_prop
    assert axes_actor.x_tip_prop is new_prop
    axes_actor.y_tip_prop = new_prop
    assert axes_actor.y_tip_prop is new_prop
    axes_actor.z_tip_prop = new_prop
    assert axes_actor.z_tip_prop is new_prop

    # test again that actors and props are stored correctly
    assert actors[0].GetProperty() is axes_actor._actor_properties[0]
    assert actors[1].GetProperty() is axes_actor._actor_properties[1]
    assert actors[2].GetProperty() is axes_actor._actor_properties[2]
    assert actors[3].GetProperty() is axes_actor._actor_properties[3]
    assert actors[4].GetProperty() is axes_actor._actor_properties[4]
    assert actors[5].GetProperty() is axes_actor._actor_properties[5]

    assert axes_actor.GetXAxisShaftProperty() is axes_actor.x_shaft_prop
    assert axes_actor.GetYAxisShaftProperty() is axes_actor.y_shaft_prop
    assert axes_actor.GetZAxisShaftProperty() is axes_actor.z_shaft_prop
    assert axes_actor.GetXAxisTipProperty() is axes_actor.x_tip_prop
    assert axes_actor.GetYAxisTipProperty() is axes_actor.y_tip_prop
    assert axes_actor.GetZAxisTipProperty() is axes_actor.z_tip_prop

    msg = "Prop must have type <class 'pyvista.plotting._property.Property'>, got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        axes_actor.x_shaft_prop = 0


def test_axes_actor_theme(axes_actor):
    assert axes_actor.x_color[0].name == 'tomato'
    assert axes_actor.x_color[1].name == 'tomato'
    assert axes_actor.y_color[0].name == 'seagreen'
    assert axes_actor.y_color[1].name == 'seagreen'
    assert axes_actor.z_color[0].name == 'mediumblue'
    assert axes_actor.z_color[1].name == 'mediumblue'

    assert axes_actor.shaft_type == 'cylinder'
    assert axes_actor.tip_type == 'cone'

    pv.global_theme.axes.x_color = 'black'
    pv.global_theme.axes.y_color = 'white'
    pv.global_theme.axes.z_color = 'gray'
    pv.global_theme.axes.shaft_type = 'line'
    pv.global_theme.axes.tip_type = 'sphere'

    axes_actor = pv.AxesActor()
    assert axes_actor.x_color[0].name == 'black'
    assert axes_actor.x_color[1].name == 'black'
    assert axes_actor.y_color[0].name == 'white'
    assert axes_actor.y_color[1].name == 'white'
    assert axes_actor.z_color[0].name == 'gray'
    assert axes_actor.z_color[1].name == 'gray'

    assert axes_actor.shaft_type == 'line'
    assert axes_actor.tip_type == 'sphere'

    # restore values
    pv.global_theme.axes.x_color = 'tomato'
    pv.global_theme.axes.y_color = 'seagreen'
    pv.global_theme.axes.z_color = 'mediumblue'
    pv.global_theme.axes.shaft_type = 'cylinder'
    pv.global_theme.axes.tip_type = 'cone'


def test_axes_actor_center(axes_actor):
    assert axes_actor.center == (0, 0, 0)
    assert axes_actor.GetCenter() == (0, 0, 0)

    axes_actor.position = (1, 2, 3)
    assert axes_actor.center == axes_actor.position
    assert axes_actor.GetCenter() == axes_actor.position

    # test center is always the origin when workaround is disabled
    axes_actor._make_orientable = False
    assert axes_actor.GetCenter() == (0, 0, 0)


@pytest.mark.parametrize('use_scale', [True, False])
def test_axes_actor_length(axes_actor, use_scale):
    default_length = 3.4641016151377544
    assert np.allclose(axes_actor.length, default_length)
    assert np.allclose(axes_actor.GetLength(), default_length)

    scaled_length = 7.4833147735478835
    if use_scale:
        axes_actor.scale = (1, 2, 3)
    else:
        axes_actor.total_length = (1, 2, 3)
    assert np.allclose(axes_actor.length, scaled_length)
    assert np.allclose(axes_actor.GetLength(), scaled_length)

    axes_actor._make_orientable = False
    if use_scale:
        # test length is not correct when workaround is disabled
        assert np.allclose(axes_actor.length, default_length)
        assert np.allclose(axes_actor.GetLength(), default_length)
    else:
        assert np.allclose(axes_actor.length, scaled_length)
        assert np.allclose(axes_actor.GetLength(), scaled_length)


def test_axes_actor_symmetric_bounds(axes_actor):
    default_bounds = (-1, 1, -1, 1, -1, 1)
    default_length = 3.4641016151377544
    assert np.allclose(axes_actor.center, (0, 0, 0))
    assert np.allclose(axes_actor.length, default_length)
    assert np.allclose(axes_actor.bounds, default_bounds)

    # test radius > length
    axes_actor.shaft_radius = 2
    assert np.allclose(axes_actor.center, (0, 0, 0))
    assert np.allclose(axes_actor.length, 5.542562584220408)
    assert np.allclose(axes_actor.bounds, (-1.6, 1.6, -1.6, 1.6, -1.6, 1.6))

    axes_actor.symmetric_bounds = False

    # make axes geometry tiny to approximate lines and points
    axes_actor.shaft_radius = 1e-8
    axes_actor.tip_radius = 1e-8
    axes_actor.shaft_length = 1
    axes_actor.tip_length = 0

    assert np.allclose(axes_actor.center, (0.5, 0.5, 0.5))
    assert np.allclose(axes_actor.length, default_length / 2)
    assert np.allclose(axes_actor.bounds, (0, 1, 0, 1, 0, 1))


# @pytest.mark.parametrize('tip_type', ['sphere', 'cone'])
# @pytest.mark.parametrize('radius', [(0.01, 0.4), (0.01, 0.4)])  # ,(1, 0),(0, 1)])
# @pytest.mark.parametrize('shaft_length', [[0.8, 0.8, 0.8]])
# @pytest.mark.parametrize('total_length', [[1, 1, 1], [4, 3, 2], [0.4, 0.5, 1.1]])
# @pytest.mark.parametrize('scale', [1, 2, 0.5])  # , [0.1, 0.2, 0.3], [2, 3, 4]])
# @pytest.mark.parametrize('auto_length', [True, False])  # ,False])
# def test_axes_actor_true_to_scale(tip_type, radius, shaft_length, total_length, scale, auto_length):
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
#     # axes_actor = AxesActor(
#     #     true_to_scale=False,
#     #     **kwargs
#     # )
#     # bounds = [actor.GetBounds() for actor in axes_actor._actors]
#     # normal_size = np.array([[b[1]-b[0],b[3]-b[2], b[5]-b[4]] for b in bounds])
#     # shaft_scale = normal_size[:3].copy()
#     # shaft_scale[np.invert(np.eye(3,dtype=bool))] /= (shaft_radius * 2)
#     #
#     # tip_scale = normal_size[3:6].copy()
#     # tip_scale[np.invert(np.eye(3,dtype=bool))] /= (tip_radius * 2)
#
#     # Create test
#     axes_actor = AxesActor(true_to_scale=True, **kwargs)
#     # bounds = [actor.GetBounds() for actor in axes_actor_true._actors]
#     # true_size = np.array([[b[1]-b[0],b[3]-b[2], b[5]-b[4]] for b in bounds])
#
#     #
#     shaft_length_scaled = np.array(shaft_length) * np.array(total_length) * np.array(scale)
#     tip_length_scaled = np.array(axes_actor.tip_length) * np.array(total_length) * np.array(scale)
#
#     # actual_scale = axes_actor._compute_true_to_scale_factors()
#     # assert actual_scale.x_shaft * shaft_length_scaled[0] == 1.0
#     # assert actual_scale.y_shaft * shaft_length_scaled[1] == 1.0
#     # assert actual_scale.z_shaft * shaft_length_scaled[2] == 1.0
#     # assert actual_scale.x_tip * tip_length_scaled[0] == 1.0
#     # assert actual_scale.y_tip * tip_length_scaled[1] == 1.0
#     # assert actual_scale.z_tip * tip_length_scaled[2] == 1.0
#
#     # axes_actor._update_props()
#     xsl, ysl, zsl = shaft_length_scaled
#     xtl, ytl, ztl = tip_length_scaled
#     sr, tr = shaft_radius * scale, tip_radius * scale
#
#     # test x shaft
#     expected_bounds = (0.0, xsl, -sr, sr, -sr, sr)
#     actual_bounds = axes_actor._actors.x_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test y shaft
#     expected_bounds = (-sr, sr, 0, ysl, -sr, sr)
#     actual_bounds = axes_actor._actors.y_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test z shaft
#     expected_bounds = (-sr, sr, -sr, sr, 0, zsl)
#     actual_bounds = axes_actor._actors.z_shaft.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds)
#
#     # test x tip
#     expected_bounds = (xsl, xsl + xtl, -tr, tr, -tr, tr)
#     actual_bounds = axes_actor._actors.x_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)
#
#     # test y tip
#     expected_bounds = (-tr, tr, ysl, ysl + ytl, -tr, tr)
#     actual_bounds = axes_actor._actors.y_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)
#
#     # test z tip
#     expected_bounds = (-tr, tr, -tr, tr, zsl, zsl + ztl)
#     actual_bounds = axes_actor._actors.z_tip.GetBounds()
#     assert np.allclose(actual_bounds, expected_bounds, rtol=0.01)


@pytest.mark.parametrize(
    'test_prop_other_prop',
    [('shaft_length', 'tip_length'), ('tip_length', 'shaft_length')],
)
@pytest.mark.parametrize('decimals', list(range(8)))
@pytest.mark.parametrize('auto_length', [True, False])
def test_axes_actor_auto_length(test_prop_other_prop, decimals, auto_length):
    test_prop, other_prop = test_prop_other_prop

    # Get default value
    axes_actor = AxesActor()
    other_prop_default = np.array(getattr(axes_actor, other_prop))

    # Initialize actor with a random length for test prop
    random_length = np.round(np.random.default_rng().random(), decimals=decimals)
    var_kwargs = {}
    var_kwargs[test_prop] = random_length
    axes_actor = AxesActor(auto_length=auto_length, **var_kwargs)

    actual_test_prop = np.array(getattr(axes_actor, test_prop))
    actual_other_prop = np.array(getattr(axes_actor, other_prop))

    expected = np.array([random_length] * 3)
    assert np.array_equal(actual_test_prop, expected)
    if auto_length:
        # Test lengths sum to 1
        actual = actual_test_prop + actual_other_prop
        expected = (1, 1, 1)
        assert np.array_equal(actual, expected)
    else:
        # Test other length is unchanged
        actual = actual_other_prop
        expected = other_prop_default
        assert np.array_equal(actual, expected)

    # Test setting both does not raise error if they sum to one
    _ = AxesActor(shaft_length=random_length, tip_length=1 - random_length)
    assert axes_actor
    _ = AxesActor(shaft_length=1 - random_length, tip_length=random_length)
    assert axes_actor

    # test enabling auto_length after object has been created
    axes_actor.auto_length = True
    setattr(axes_actor, test_prop, 0.9)
    expected = (0.9, 0.9, 0.9)
    actual = getattr(axes_actor, test_prop)
    assert np.array_equal(actual, expected)

    value_unchanged = (0.1, 0.1, 0.1)
    expected = value_unchanged
    actual = getattr(axes_actor, other_prop)
    assert np.array_equal(actual, expected)

    # test disabling auto_length after object has been created
    axes_actor.auto_length = False
    setattr(axes_actor, test_prop, 0.7)
    expected = (0.7, 0.7, 0.7)
    actual = getattr(axes_actor, test_prop)
    assert np.array_equal(actual, expected)

    expected = value_unchanged
    actual = getattr(axes_actor, other_prop)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize('enabled', [True, False])
def test_axes_actor_auto_shaft_type(enabled):
    axes_actor = AxesActor(auto_shaft_type=enabled)

    assert axes_actor.shaft_type == 'cylinder'

    axes_actor.shaft_width = 5
    if enabled:
        assert axes_actor.shaft_type == 'line'
    else:
        assert axes_actor.shaft_type == 'cylinder'
        axes_actor.shaft_type = 'line'

    axes_actor.shaft_resolution = 2
    if enabled:
        assert axes_actor.shaft_type == 'cylinder'
    else:
        assert axes_actor.shaft_type == 'line'
        axes_actor.shaft_type = 'cylinder'

    # set up for next test
    axes_actor.shaft_type = 'line'
    assert axes_actor.shaft_type == 'line'

    axes_actor.shaft_radius = 1
    if enabled:
        assert axes_actor.shaft_type == 'cylinder'
    else:
        assert axes_actor.shaft_type == 'line'


def test_axes_actor_auto_length_raises():
    msg = (
        "Cannot set both `shaft_length` and `tip_length` when `auto_length` is `True`.\n"
        "Set either `shaft_length` or `tip_length`, but not both."
    )
    with pytest.raises(ValueError, match=msg):
        AxesActor(shaft_length=0.6, tip_length=0.6, auto_length=True)


def test_axes_actor_auto_shaft_type_raises():
    msg = (
        "Cannot set `shaft_width` when type is 'cylinder' and `auto_shaft_type=True`.\n"
        "Only `shaft_radius` or `shaft_resolution` can be set."
    )
    with pytest.raises(ValueError, match=msg):
        AxesActor(shaft_type='cylinder', shaft_width=5)

    msg = (
        "Cannot set properties `shaft_radius` or `shaft_resolution` when shaft type is 'line'\n"
        "and `auto_shaft_type=True`. Only `shaft_width` can be set."
    )

    with pytest.raises(ValueError, match=msg):
        AxesActor(shaft_type='line', shaft_radius=1)
    with pytest.raises(ValueError, match=msg):
        AxesActor(shaft_type='line', shaft_resolution=1)

    msg = (
        'Cannot set line properties (`shaft_width`) and cylinder properties (`shaft_radius`\n'
        'or `shaft_resolution`) simultaneously when`auto_shaft_type=True`.'
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        AxesActor(shaft_resolution=1, shaft_width=5)


def test_axes_actor_composite():
    axes = pv.AxesActorComposite()
    assert axes
