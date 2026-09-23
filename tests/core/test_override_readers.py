"""Tests for built-in formats a companion package overrides."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities import reader_registry as _reg_mod

OVERRIDES = (
    pytest.param('.stl', 'pyvista_stl:read_as_mesh', pv.STLReader, id='stl'),
    pytest.param('.ply', 'pyvista_miniply:read_as_mesh', pv.PLYReader, id='ply'),
)
EXTS = tuple(pytest.param(p.values[0], id=p.id) for p in OVERRIDES)
EXT_BUILTINS = tuple(pytest.param(p.values[0], p.values[2], id=p.id) for p in OVERRIDES)
EXT_SOURCES = tuple(pytest.param(p.values[0], p.values[1], id=p.id) for p in OVERRIDES)


@pytest.fixture(autouse=True)
def _clean_registry():
    state = _reg_mod._save_registry_state()
    yield
    _reg_mod._restore_registry_state(state)


@pytest.fixture
def stl_file(tmp_path):
    path = tmp_path / 'mesh.stl'
    pv.Sphere().save(path)
    return str(path)


@pytest.fixture
def ply_file(tmp_path):
    path = tmp_path / 'mesh.ply'
    pv.Sphere().save(path)
    return str(path)


@pytest.mark.parametrize(('ext', 'source'), EXT_SOURCES)
def test_package_claims_the_extension(ext, source):
    assert ext in pv.core.utilities.reader.CLASS_READERS
    _reg_mod._get_ext_handler(ext)

    registration = next(r for r in pv.registered_readers() if r.extension == ext)
    assert registration.source == source
    assert registration.override
    assert not registration.reader_class


@pytest.mark.parametrize(('ext', 'builtin'), EXT_BUILTINS)
def test_read_matches_the_builtin_reader(ext, builtin, request):
    path = request.getfixturevalue(f'{ext.lstrip(".")}_file')

    fast = pv.read(path)
    reference = builtin(path).read()

    assert fast.n_points == reference.n_points
    assert fast.n_cells == reference.n_cells
    assert np.allclose(fast.points, reference.points)
    assert np.array_equal(fast.regular_faces, reference.regular_faces)
    assert fast.area == pytest.approx(reference.area)


@pytest.mark.parametrize(('ext', 'builtin'), EXT_BUILTINS)
def test_get_reader_still_returns_the_builtin_class(ext, builtin, request):
    path = request.getfixturevalue(f'{ext.lstrip(".")}_file')

    # The packages register a callable, so the class path is unaffected.
    assert isinstance(pv.get_reader(path), builtin)


@pytest.mark.parametrize('ext', EXTS)
def test_explicit_registration_beats_the_package(ext, request):
    path = request.getfixturevalue(f'{ext.lstrip(".")}_file')
    sentinel = pv.PolyData()

    pv.register_reader(ext, lambda _path, **_kwargs: sentinel, override=True)

    assert pv.read(path) is sentinel


def test_ply_keeps_point_arrays(tmp_path):
    mesh = pv.Sphere()
    mesh.point_data['RGB'] = np.tile(np.uint8([10, 20, 30]), (mesh.n_points, 1))
    path = tmp_path / 'colored.ply'
    mesh.save(path, texture='RGB')

    fast = pv.read(path)

    assert 'RGB' in fast.point_data
    assert np.array_equal(fast['RGB'], mesh['RGB'])


def test_airplane_example_reads_through_the_override():
    mesh = pv.read(examples.planefile)

    assert mesh.n_points == 1335
    assert mesh.n_cells == 2452


@pytest.mark.parametrize('ext', EXTS)
def test_reader_arguments_fall_back_to_the_builtin(ext, request):
    path = request.getfixturevalue(f'{ext.lstrip(".")}_file')

    with patch.object(pv.BaseReader, 'show_progress') as show_progress:
        pv.read(path, progress_bar=True)

    show_progress.assert_called_once()


@pytest.mark.parametrize(('ext', 'builtin'), EXT_BUILTINS)
def test_validate_falls_back_to_the_builtin(ext, builtin, request):
    path = request.getfixturevalue(f'{ext.lstrip(".")}_file')

    assert pv.read(path, validate=True) == builtin(path).read()
