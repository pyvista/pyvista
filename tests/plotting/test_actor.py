from __future__ import annotations

import numpy as np
import pytest
import scipy
import vtk

import pyvista as pv
from pyvista import examples
from pyvista.plotting.prop3d import Prop3D
from pyvista.plotting.prop3d import _orientation_as_rotation_matrix
from pyvista.plotting.prop3d import _Prop3DMixin
from pyvista.plotting.prop3d import _rotation_matrix_as_orientation


@pytest.fixture
def actor():
    return pv.Plotter().add_mesh(pv.Plane())


@pytest.fixture
def volume():
    return pv.Plotter().add_volume(pv.Wavelet())


@pytest.fixture
def actor_from_multi_block():
    return pv.Plotter().add_mesh(pv.MultiBlock([pv.Plane()]))


@pytest.fixture
def dummy_actor(actor):
    # Define prop3d-like class
    class DummyActor(_Prop3DMixin):
        def __init__(self):
            super().__init__()
            self._actor = actor

        def _post_set_update(self):
            # Apply the same transformation to the underlying actor
            self._actor.user_matrix = self._transformation_matrix

        def _get_bounds(self):
            return self._actor.GetBounds()

    # Sanity checks to make sure fixture is defined properly
    dummy_actor = DummyActor()
    assert not isinstance(dummy_actor, Prop3D)
    assert isinstance(dummy_actor, _Prop3DMixin)
    assert dummy_actor.bounds == actor.GetBounds()
    return dummy_actor


@pytest.fixture
def vol_actor():
    vol = pv.ImageData(dimensions=(10, 10, 10))
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

    assert actor.memory_address == actor.GetAddressAsString('')

    actor.user_matrix = None
    repr_ = repr(actor)
    assert 'User matrix:                Identity' in repr_

    actor.user_matrix = np.eye(4) * 2
    repr_ = repr(actor)
    assert 'User matrix:                Set' in repr_


def test_actor_from_argument():
    mapper = pv.DataSetMapper()
    prop = pv.Property()
    name = 'Actor'
    actor = pv.Actor(mapper=mapper, prop=prop, name=name)
    assert actor.mapper is mapper
    assert actor.prop is prop
    assert actor.name == name


def test_actor_from_plotter():
    mesh = pv.Sphere()
    actor = pv.Plotter().add_mesh(mesh, lighting=False)
    assert actor.mapper.dataset is mesh
    assert actor.prop.lighting is False
    assert 'Mapper' in repr(actor)


@pytest.mark.parametrize('include_mapper', [True, False])
@pytest.mark.parametrize(('prop3d', 'prop_attr'), [(pv.Volume, 'shade'), (pv.Actor, 'lighting')])
def test_actor_copy_deep(prop3d, prop_attr, actor, volume, include_mapper):
    obj = actor if prop3d is pv.Actor else volume
    if include_mapper:
        assert obj.mapper is not None
    else:
        obj.mapper = None
        assert obj.mapper is None

    copied = obj.copy()
    assert copied is not obj
    assert copied.prop is not obj.prop

    setattr(copied.prop, prop_attr, not getattr(copied.prop, prop_attr))
    assert getattr(copied.prop, prop_attr) is not getattr(obj.prop, prop_attr)

    if include_mapper:
        assert copied.mapper is not obj.mapper
        assert copied.mapper.dataset is obj.mapper.dataset
        copied.mapper.dataset = None
        assert obj.mapper.dataset is not None
    else:
        assert copied.mapper is None


@pytest.mark.parametrize('prop3d', [pv.Volume, pv.Actor])
def test_actor_copy_shallow(prop3d, actor, volume):
    obj = actor if prop3d is pv.Actor else volume
    copied = obj.copy(deep=False)
    assert copied is not obj
    assert copied.prop is obj.prop
    assert copied.mapper is obj.mapper


def test_actor_mblock_copy_shallow(actor_from_multi_block):
    actor_copy = actor_from_multi_block.copy(deep=False)
    assert actor_copy is not actor_from_multi_block
    assert actor_copy.prop is actor_from_multi_block.prop
    assert actor_copy.mapper is actor_from_multi_block.mapper
    assert actor_copy.mapper.dataset is actor_from_multi_block.mapper.dataset


@pytest.mark.skip_mac('MacOS CI fails when downloading examples')
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


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_scale(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert actor.scale == (1, 1, 1)
    scale = (2, 2, 2)
    actor.scale = scale
    assert actor.scale == scale
    actor.scale = 3
    assert actor.scale == (3, 3, 3)


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_position(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
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


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_orientation(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert actor.orientation == (0, 0, 0)
    orientation = (10, 20, 30)
    actor.orientation = orientation
    assert np.allclose(actor.orientation, orientation)


def test_actor_rotation_order(actor):
    orientation = (10, 20, 30)
    dataset = pv.Cube()
    dataset.rotate_y(orientation[1], inplace=True)
    dataset.rotate_x(orientation[0], inplace=True)
    dataset.rotate_z(orientation[2], inplace=True)

    actor = pv.Actor(mapper=pv.DataSetMapper(pv.Cube()))
    actor.orientation = orientation
    assert np.allclose(dataset.bounds, actor.bounds)


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_origin(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert actor.origin == (0, 0, 0)
    origin = (1, 2, 3)
    actor.origin = origin
    assert np.allclose(actor.origin, origin)


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_length(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    initial_length = 2**0.5  # sqrt(2)
    scale_factor = 2

    assert actor.length == initial_length
    actor.scale = scale_factor
    assert actor.length == initial_length * scale_factor


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_user_matrix(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert np.allclose(actor.user_matrix, np.eye(4))

    arr = np.array(
        [[0.707, -0.707, 0, 0], [0.707, 0.707, 0, 0], [0, 0, 1, 1.500001], [0, 0, 0, 2]]
    )

    actor.user_matrix = arr
    assert isinstance(actor.user_matrix, np.ndarray)
    assert np.allclose(actor.user_matrix, arr)


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_bounds(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert isinstance(actor.bounds, tuple)
    assert np.allclose(actor.bounds, (-0.5, 0.5, -0.5, 0.5, 0, 0))


@pytest.mark.parametrize('klass', ['Prop3D, Prop3DMixin'])
def test_actor_center(klass, actor, dummy_actor):
    actor = actor if klass == 'Prop3D' else dummy_actor
    assert actor.center == (0.0, 0.0, 0.0)


def test_actor_name(actor):
    actor.name = 1
    assert actor._name == '1'

    with pytest.raises(ValueError, match='Name must be truthy'):
        actor.name = None


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


@pytest.mark.parametrize(
    'func',
    [np.array, scipy.spatial.transform.Rotation.from_matrix],
    ids=['numpy', 'scipy'],
)
def test_rotation_from(actor, func):
    array = [
        [0.78410209, -0.49240388, 0.37778609],
        [0.52128058, 0.85286853, 0.02969559],
        [-0.33682409, 0.17364818, 0.92541658],
    ]

    rotation = func(array)
    actor.rotation_from(rotation)

    expected = (10, 20, 30)
    actual = actor.orientation
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('origin', [None, [1, 2, 3]])
def test_rotation_from_matches_dataset_rotate(origin):
    array = [
        [0.78410209, -0.49240388, 0.37778609],
        [0.52128058, 0.85286853, 0.02969559],
        [-0.33682409, 0.17364818, 0.92541658],
    ]
    # Rotate dataset and actor independently
    dataset = pv.Cube()
    actor = pv.Actor(mapper=pv.DataSetMapper(dataset=pv.Cube()))
    dataset.rotate(array, point=origin, inplace=True)
    actor.rotation_from(array)
    if origin:
        actor.origin = origin
    assert np.allclose(dataset.bounds, actor.bounds)


@pytest.mark.parametrize('order', ['F', 'C'])
def test_convert_orientation_to_rotation_matrix(order):
    orientation = (10, 20, 30)
    rotation = np.array(
        [
            [0.78410209, -0.49240388, 0.37778609],
            [0.52128058, 0.85286853, 0.02969559],
            [-0.33682409, 0.17364818, 0.92541658],
        ],
        order=order,
    )

    actual_rotation = _orientation_as_rotation_matrix(orientation)
    assert isinstance(actual_rotation, np.ndarray)
    assert actual_rotation.shape == (3, 3)
    assert np.allclose(actual_rotation, rotation)

    actual_orientation = _rotation_matrix_as_orientation(rotation)
    assert isinstance(actual_orientation, tuple)
    assert len(actual_orientation) == 3
    assert np.allclose(actual_orientation, orientation)


@pytest.mark.parametrize('multiply_mode', ['pre', 'post'])
def test_transform_actor(actor, multiply_mode):
    translation = pv.Transform().translate((1, 2, 3))
    scaling = pv.Transform().scale(2)

    expected = pv.Transform([translation, scaling], multiply_mode=multiply_mode)

    actor1 = actor.transform(translation, multiply_mode=multiply_mode, inplace=True)
    assert actor1 is actor
    actor2 = actor1.transform(scaling, multiply_mode=multiply_mode, inplace=False)
    assert actor2 is not actor1

    assert np.allclose(actor2.user_matrix, expected.matrix)


def test_follower():
    mesh = pv.Sphere()
    mapper = pv.DataSetMapper(mesh)
    follower = pv.Follower(mapper=mapper)
    camera = pv.Camera()
    follower.camera = camera
    assert follower.mapper is mapper
    assert follower.prop is not None
    assert follower.camera is camera
