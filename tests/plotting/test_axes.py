from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import _vtk
from pyvista.plotting.opts import InterpolationType
from pyvista.plotting.opts import RepresentationType

# A large number of tests here fail gc
pytestmark = pytest.mark.skip_check_gc


@pytest.fixture
def axes():
    return pv.Axes()


@pytest.fixture
def axes_actor(axes):
    return axes.axes_actor


def test_actors(axes):
    actor = axes.actor

    # test showing
    assert not actor.GetVisibility()
    axes.show_actor()
    assert actor.GetVisibility()

    # test hiding
    assert actor.GetVisibility()
    axes.hide_actor()
    assert not actor.GetVisibility()


def test_origin(axes):
    origin = np.random.default_rng().random(3)
    axes.origin = origin
    assert np.all(axes.GetOrigin() == origin)
    assert np.all(axes.origin == origin)


def test_symmetric(axes):
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


def test_axes_actor_total_len(axes_actor):
    axes_actor.total_length = 1
    assert axes_actor.total_length == (1, 1, 1)

    axes_actor.total_length = (1, 2, 3)
    assert axes_actor.total_length == (1, 2, 3)


def test_axes_actor_shaft_len(axes_actor):
    axes_actor.shaft_length = 1
    assert axes_actor.shaft_length == (1, 1, 1)

    axes_actor.shaft_length = (1, 2, 3)
    assert axes_actor.shaft_length == (1, 2, 3)


def test_axes_actor_tip_len(axes_actor):
    axes_actor.tip_length = 1
    assert axes_actor.tip_length == (1, 1, 1)

    axes_actor.tip_length = (1, 2, 3)
    assert axes_actor.tip_length == (1, 2, 3)


def test_axes_actor_label_pos(axes_actor):
    axes_actor.label_position = 1
    assert axes_actor.label_position == (1, 1, 1)

    axes_actor.label_position = (1, 2, 3)
    assert axes_actor.label_position == (1, 2, 3)


def test_axes_actor_cone_res(axes_actor):
    axes_actor.cone_resolution = 24
    assert axes_actor.cone_resolution == 24


def test_axes_actor_sphere_res(axes_actor):
    axes_actor.sphere_resolution = 24
    assert axes_actor.sphere_resolution == 24


def test_axes_actor_cylinder_res(axes_actor):
    axes_actor.cylinder_resolution = 24
    assert axes_actor.cylinder_resolution == 24


def test_axes_actor_cone_rad(axes_actor):
    axes_actor.cone_radius = 0.8
    assert axes_actor.cone_radius == 0.8


def test_axes_actor_sphere_rad(axes_actor):
    axes_actor.sphere_radius = 0.8
    assert axes_actor.sphere_radius == 0.8


def test_axes_actor_cylinder_rad(axes_actor):
    axes_actor.cylinder_radius = 0.03
    assert axes_actor.cylinder_radius == 0.03


def test_axes_actor_shaft_type(axes_actor):
    axes_actor.shaft_type = pv.AxesActor.ShaftType.CYLINDER
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.CYLINDER
    axes_actor.shaft_type = pv.AxesActor.ShaftType.LINE
    assert axes_actor.shaft_type == pv.AxesActor.ShaftType.LINE


def test_axes_actor_tip_type(axes_actor):
    axes_actor.tip_type = pv.AxesActor.TipType.CONE
    assert axes_actor.tip_type == pv.AxesActor.TipType.CONE
    axes_actor.tip_type = pv.AxesActor.TipType.SPHERE
    assert axes_actor.tip_type == pv.AxesActor.TipType.SPHERE


def test_axes_actor_labels_individual(axes_actor):
    axes_actor.x_label = 'Axis X'
    axes_actor.y_label = 'Axis Y'
    axes_actor.z_label = 'Axis Z'

    assert axes_actor.x_label == 'Axis X'
    assert axes_actor.y_label == 'Axis Y'
    assert axes_actor.z_label == 'Axis Z'


def test_axes_actor_labels_group(axes_actor):
    new_labels = ['label1', 'label2', 'label3']
    axes_actor.labels = new_labels
    assert axes_actor.labels == tuple(new_labels)
    assert axes_actor.x_label == new_labels[0]
    assert axes_actor.y_label == new_labels[1]
    assert axes_actor.z_label == new_labels[2]

    match = 'Labels must be a list or tuple. Got abc instead.'
    with pytest.raises(TypeError, match=match):
        axes_actor.labels = 'abc'

    match = "Labels must be a list or tuple with three items. Got ['1', '2'] instead."
    with pytest.raises(ValueError, match=re.escape(match)):
        axes_actor.labels = ['1', '2']


def test_axes_actor_properties():
    prop = pv.ActorProperties(_vtk.vtkProperty())

    prop.color = (1, 1, 1)
    assert prop.color == (1, 1, 1)

    prop.metallic = 0.2
    assert prop.metallic == 0.2

    prop.roughness = 0.3
    assert prop.roughness == 0.3

    prop.anisotropy = 0.4
    assert prop.anisotropy == 0.4

    prop.anisotropy_rotation = 0.4
    assert prop.anisotropy_rotation == 0.4

    prop.lighting = False
    assert not prop.lighting

    prop.interpolation_model = InterpolationType.PHONG
    assert prop.interpolation_model == InterpolationType.PHONG

    prop.index_of_refraction = 1.5
    assert prop.index_of_refraction == 1.5

    prop.opacity = 0.6
    assert prop.opacity == 0.6

    prop.shading = False
    assert not prop.shading

    prop.representation = RepresentationType.POINTS
    assert prop.representation == RepresentationType.POINTS
