import platform

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples

skip_mac = pytest.mark.skipif(
    platform.system() == 'Darwin', reason='MacOS CI fails when downloading examples'
)


@pytest.fixture()
def actor():
    return pv.Plotter().add_mesh(pv.Plane())


@pytest.fixture()
def vol_actor():
    vol = pv.UniformGrid(dimensions=(10, 10, 10))
    vol['scalars'] = 255 - vol.z * 25
    pl = pv.Plotter()
    return pl.add_volume(vol)


def test_actor_init_empty():
    actor = pv.Actor()
    assert 'Position' in repr(actor)
    assert 'Has mapper' in repr(actor)
    assert 'Mapper' not in repr(actor)
    assert isinstance(actor.prop, pv.Property)
    assert actor.mapper is None

    with pytest.raises(AttributeError):
        actor.not_an_attribute = None

    assert actor.memory_address == actor.GetAddressAsString("")


def test_actor_from_plotter():
    mesh = pv.Sphere()
    actor = pv.Plotter().add_mesh(mesh, lighting=False)
    assert actor.mapper.dataset is mesh
    assert actor.prop.lighting is False
    assert 'Mapper' in repr(actor)


def test_actor_copy_deep(actor):
    actor_copy = actor.copy()
    assert actor_copy is not actor

    assert actor_copy.prop is not actor.prop
    actor_copy.prop.lighting = not actor_copy.prop.lighting
    assert actor_copy.prop.lighting is not actor.prop.lighting

    assert actor_copy.mapper is not actor.mapper
    assert actor_copy.mapper.dataset is actor.mapper.dataset
    actor_copy.mapper.dataset = None
    assert actor.mapper.dataset is not None


def test_actor_copy_shallow(actor):
    actor_copy = actor.copy(deep=False)
    assert actor_copy is not actor
    assert actor_copy.prop is actor.prop
    assert actor_copy.mapper is actor.mapper


@skip_mac
def test_actor_texture(actor):
    texture = examples.download_masonry_texture()
    actor.texture = texture
    assert actor.texture is texture


def test_actor_pickable(actor):
    actor.pickable = True
    assert actor.pickable is True


def test_actor_visible(actor):
    actor.visibility = True
    assert actor.visibility is True


def test_actor_scale(actor):
    assert actor.scale == (1, 1, 1)
    scale = (2, 2, 2)
    actor.scale = scale
    assert actor.scale == scale


def test_actor_position(actor):
    assert actor.position == (0, 0, 0)
    position = (2, 2, 2)
    actor.position = position
    assert actor.position == position


def test_actor_rotate_x(actor):
    actor.rotate_x(90)
    assert np.allclose(actor.orientation, (90, 0, 0))


def test_actor_rotate_y(actor):
    actor.rotate_y(90)
    assert np.allclose(actor.orientation, (0, 90, 0))


def test_actor_rotate_z(actor):
    actor.rotate_z(90)
    assert np.allclose(actor.orientation, (0, 0, 90))


def test_actor_orientation(actor):
    actor.orientation == (0, 0, 0)
    orientation = (10, 20, 30)
    actor.orientation = orientation
    assert np.allclose(actor.orientation, orientation)


def test_actor_unit_matrix(actor):
    assert actor.user_matrix is None

    arr = np.array([[0.707, -0.707, 0, 0], [0.707, 0.707, 0, 0], [0, 0, 1, 1.500001], [0, 0, 0, 2]])

    actor.user_matrix = arr
    assert isinstance(actor.user_matrix, np.ndarray)
    assert np.allclose(actor.user_matrix, arr)


def test_actor_bounds(actor):
    assert isinstance(actor.bounds, tuple)
    assert np.allclose(actor.bounds, (-0.5, 0.5, -0.5, 0.5, 0, 0))


def test_actor_center(actor):
    assert actor.center == (0.0, 0.0, 0.0)


def test_actor_backface_prop(actor):
    actor.prop.opacity = 0.5
    assert isinstance(actor.backface_prop, pv.Property)
    assert actor.backface_prop.opacity == actor.prop.opacity
    actor.backface_prop.opacity = 1.0
    assert actor.backface_prop.opacity == 1.0

    actor.backface_prop = None
    assert actor.backface_prop.opacity == actor.prop.opacity


def test_vol_actor_prop(vol_actor):
    assert isinstance(vol_actor.prop, vtk.vtkVolumeProperty)

    prop = vtk.vtkVolumeProperty()
    vol_actor.prop = prop
    assert vol_actor.prop is prop
