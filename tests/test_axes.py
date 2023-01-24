import numpy as np

import pyvista


def test_actors():
    axes = pyvista.Axes()
    actor = axes.actor

    # test showing
    assert not actor.GetVisibility()
    axes.show_actor()
    assert actor.GetVisibility()

    # test hiding
    assert actor.GetVisibility()
    axes.hide_actor()
    assert not actor.GetVisibility()


def test_origin():
    axes = pyvista.Axes()

    origin = np.random.random(3)
    axes.origin = origin
    assert np.all(axes.GetOrigin() == origin)
    assert np.all(axes.origin == origin)


def test_symmetric():
    axes = pyvista.Axes()

    # test showing
    assert not axes.GetSymmetric()
    axes.show_symmetric()
    assert axes.GetSymmetric()

    # test hiding
    assert axes.GetSymmetric()
    axes.hide_symmetric()
    assert not axes.GetSymmetric()


def test_axes_actor_visibility():
    axes = pyvista.Axes()
    assert axes.axes_actor.visibility
    axes.axes_actor.visibility = False
    assert not axes.axes_actor.visibility


def test_axes_actor_total_len():
    axes = pyvista.Axes()
    axes.axes_actor.total_length = 1
    assert axes.axes_actor.total_length == (1, 1, 1)

    axes.axes_actor.total_length = (1, 2, 3)
    assert axes.axes_actor.total_length == (1, 2, 3)


def test_axes_actor_shaft_len():
    axes = pyvista.Axes()
    axes.axes_actor.shaft_length = 1
    assert axes.axes_actor.shaft_length == (1, 1, 1)

    axes.axes_actor.shaft_length = (1, 2, 3)
    assert axes.axes_actor.shaft_length == (1, 2, 3)


def test_axes_actor_tip_len():
    axes = pyvista.Axes()
    axes.axes_actor.tip_length = 1
    assert axes.axes_actor.tip_length == (1, 1, 1)

    axes.axes_actor.tip_length = (1, 2, 3)
    assert axes.axes_actor.tip_length == (1, 2, 3)


def test_axes_actor_label_pos():
    axes = pyvista.Axes()
    axes.axes_actor.label_position = 1
    assert axes.axes_actor.label_position == (1, 1, 1)

    axes.axes_actor.label_position = (1, 2, 3)
    assert axes.axes_actor.label_position == (1, 2, 3)


def test_axes_actor_cone_res():
    axes = pyvista.Axes()
    axes.axes_actor.cone_resolution = 24
    assert axes.axes_actor.cone_resolution == 24


def test_axes_actor_sphere_res():
    axes = pyvista.Axes()
    axes.axes_actor.sphere_resolution = 24
    assert axes.axes_actor.sphere_resolution == 24


def test_axes_actor_cylinder_res():
    axes = pyvista.Axes()
    axes.axes_actor.cylinder_resolution = 24
    assert axes.axes_actor.cylinder_resolution == 24


def test_axes_actor_cone_rad():
    axes = pyvista.Axes()
    axes.axes_actor.cone_radius = 0.8
    assert axes.axes_actor.cone_radius == 0.8


def test_axes_actor_sphere_rad():
    axes = pyvista.Axes()
    axes.axes_actor.sphere_radius = 0.8
    assert axes.axes_actor.sphere_radius == 0.8


def test_axes_actor_cylinder_rad():
    axes = pyvista.Axes()
    axes.axes_actor.cylinder_radius = 0.03
    assert axes.axes_actor.cylinder_radius == 0.03


def test_axes_actor_shaft_type():
    axes = pyvista.Axes()
    axes.axes_actor.shaft_type = pyvista.AxesActor.ShaftType.CYLINDER
    assert axes.axes_actor.shaft_type == pyvista.AxesActor.ShaftType.CYLINDER
    axes.axes_actor.shaft_type = pyvista.AxesActor.ShaftType.LINE
    assert axes.axes_actor.shaft_type == pyvista.AxesActor.ShaftType.LINE


def test_axes_actor_tip_type():
    axes = pyvista.Axes()
    axes.axes_actor.tip_type = pyvista.AxesActor.TipType.CONE
    assert axes.axes_actor.tip_type == pyvista.AxesActor.TipType.CONE
    axes.axes_actor.tip_type = pyvista.AxesActor.TipType.SPHERE
    assert axes.axes_actor.tip_type == pyvista.AxesActor.TipType.SPHERE


def test_axes_actor_axis_labels():
    axes = pyvista.Axes()
    axes.axes_actor.x_axis_label = 'Axis X'
    axes.axes_actor.y_axis_label = 'Axis Y'
    axes.axes_actor.z_axis_label = 'Axis Z'

    assert axes.axes_actor.x_axis_label == 'Axis X'
    assert axes.axes_actor.y_axis_label == 'Axis Y'
    assert axes.axes_actor.z_axis_label == 'Axis Z'
