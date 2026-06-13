"""Tests for core utilities that require rendering classes."""

from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.utilities.transform import Transform
from pyvista.plotting.prop3d import _orientation_as_rotation_matrix
from pyvista.plotting.widgets import _parse_interaction_event

SCALE = 2
VECTOR = (1, 2, 3)


@pytest.fixture
def transform():
    return Transform()


@pytest.fixture
def scale_transform():
    return Transform() * SCALE


@pytest.fixture
def translate_transform():
    return Transform() + VECTOR


@pytest.mark.parametrize('mode', ['replace', 'pre-multiply', 'post-multiply'])
@pytest.mark.parametrize('method', [pv.Transform.apply, pv.Transform.apply_to_actor])
def test_transform_apply_to_actor(scale_transform, translate_transform, mode, method):
    expected_matrix = scale_transform.matrix
    actor = pv.Actor()

    transformed = method(scale_transform, actor, mode)
    assert np.allclose(transformed.user_matrix, expected_matrix)

    transformed = method(translate_transform, transformed, mode)
    if mode == 'replace':
        expected_matrix = translate_transform.matrix
    else:
        expected_matrix = scale_transform.compose(
            translate_transform, multiply_mode=mode.split('-')[0]
        ).matrix
    assert np.allclose(transformed.user_matrix, expected_matrix)


def test_transform_apply_invalid_actor_mode():
    actor = pv.Actor()
    trans = pv.Transform()

    match = (
        "Transformation mode 'vectors' is not supported for actors. Mode must be one of\n"
        "['replace', 'pre-multiply', 'post-multiply', None]"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        trans.apply(actor, 'vectors')


@pytest.fixture
def transformed_actor():
    actor = pv.Actor()
    actor.position = (-0.5, -0.5, 1)
    actor.orientation = (10, 20, 30)
    actor.scale = (1.5, 2, 2.5)
    actor.origin = (2, 1.5, 1)
    actor.user_matrix = pv.array_from_vtkmatrix(actor.GetMatrix())
    return actor


@pytest.mark.parametrize('override_mode', ['pre', 'post'])
@pytest.mark.parametrize('object_mode', ['pre', 'post'])
def test_transform_multiply_mode_override(
    transform, transformed_actor, object_mode, override_mode
):
    transform.multiply_mode = object_mode

    transform.translate(np.array(transformed_actor.origin) * -1, multiply_mode=override_mode)
    transform.scale(transformed_actor.scale, multiply_mode=override_mode)
    rotation = _orientation_as_rotation_matrix(transformed_actor.orientation)
    transform.rotate(rotation, multiply_mode=override_mode)
    transform.translate(np.array(transformed_actor.origin), multiply_mode=override_mode)
    transform.translate(transformed_actor.position, multiply_mode=override_mode)
    transform.compose(transformed_actor.user_matrix, multiply_mode=override_mode)

    transform_matrix = transform.matrix
    actor_matrix = pv.array_from_vtkmatrix(transformed_actor.GetMatrix())
    if override_mode == 'post':
        assert np.allclose(transform_matrix, actor_matrix)
    else:
        assert not np.allclose(transform_matrix, actor_matrix)


@pytest.mark.parametrize(
    ('event', 'expected'),
    [
        ('end', _vtk.vtkCommand.EndInteractionEvent),
        ('start', _vtk.vtkCommand.StartInteractionEvent),
        ('always', _vtk.vtkCommand.InteractionEvent),
        (_vtk.vtkCommand.InteractionEvent,) * 2,
        (_vtk.vtkCommand.EndInteractionEvent,) * 2,
        (_vtk.vtkCommand.StartInteractionEvent,) * 2,
    ],
)
def test_parse_interaction_event(
    event: str | _vtk.vtkCommand.EventIds,
    expected: _vtk.vtkCommand.EventIds,
):
    assert _parse_interaction_event(event) == expected


def test_parse_interaction_event_raises_str():
    with pytest.raises(
        ValueError,
        match=r'Expected.*start.*end.*always.*foo was given',
    ):
        _parse_interaction_event('foo')


def test_parse_interaction_event_raises_wrong_type():
    with pytest.raises(
        TypeError,
        match=r'.*either a str or.*vtk.vtkCommand.EventIds.*int.* was given',
    ):
        _parse_interaction_event(1)
