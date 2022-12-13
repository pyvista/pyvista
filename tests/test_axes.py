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
