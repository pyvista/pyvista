"""Tests for formats served by companion packages rather than by PyVista."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import pyvista_frd

import pyvista as pv
from pyvista.core.utilities import reader_registry as _reg_mod

FRD_CONTENT = """2C
 -1    1 0.0 0.0 0.0
 -1    2 1.0 0.0 0.0
 -1    3 1.0 1.0 0.0
 -1    4 0.0 1.0 0.0
 -1    5 0.0 0.0 1.0
 -1    6 1.0 0.0 1.0
 -1    7 1.0 1.0 1.0
 -1    8 0.0 1.0 1.0
 -3
 3C
 -1    1    1
 -2    1    2    3    4    5    6    7    8
 -3
 100CL 1 0.1
 -4 DISP 3
 -1    1 1.0 2.0 3.0
 -1    2 1.0 2.0 3.0
 -1    3 1.0 2.0 3.0
 -1    4 1.0 2.0 3.0
 -1    5 1.0 2.0 3.0
 -1    6 1.0 2.0 3.0
 -1    7 1.0 2.0 3.0
 -1    8 1.0 2.0 3.0
 -3
 100CL 2 0.2
 -4 DISP 3
 -1    1 2.0 4.0 6.0
 -1    2 2.0 4.0 6.0
 -1    3 2.0 4.0 6.0
 -1    4 2.0 4.0 6.0
 -1    5 2.0 4.0 6.0
 -1    6 2.0 4.0 6.0
 -1    7 2.0 4.0 6.0
 -1    8 2.0 4.0 6.0
 -3
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    """Undo the lazy registration a ``.frd`` lookup performs."""
    state = _reg_mod._save_registry_state()
    yield
    _reg_mod._restore_registry_state(state)
    _reg_mod._entry_points_loaded = state['entry_points_loaded']


@pytest.fixture
def frd_file(tmp_path):
    path = tmp_path / 'model.frd'
    path.write_text(FRD_CONTENT, encoding='utf-8')
    return str(path)


@pytest.fixture
def frd_not_installed(monkeypatch):
    """Make ``pyvista_frd`` raise ``ModuleNotFoundError`` on import."""
    monkeypatch.setitem(sys.modules, 'pyvista_frd', None)


class _BrokenFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Report ``pyvista_frd`` as present, then fail to execute it."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        return (
            importlib.machinery.ModuleSpec(fullname, self) if fullname == 'pyvista_frd' else None
        )

    def exec_module(self, module):  # noqa: ARG002
        msg = 'libpvfrd.so: cannot open shared object file'
        raise ImportError(msg)


@pytest.fixture
def frd_install_broken(monkeypatch):
    monkeypatch.delitem(sys.modules, 'pyvista_frd', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenFinder(), *sys.meta_path])


def test_read_dispatches_to_companion_package(frd_file):
    mesh = pv.read(frd_file)

    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_points == 8
    assert mesh.n_cells == 1
    assert mesh.celltypes.tolist() == [pv.CellType.HEXAHEDRON]
    assert sorted(mesh.point_data) == ['DISP', 'original_node_ids']
    assert np.allclose(mesh['DISP'], np.tile([1.0, 2.0, 3.0], (8, 1)))


def test_read_matches_the_package_called_directly(frd_file):
    assert pv.read(frd_file) == pyvista_frd.read(frd_file)


def test_registered_as_a_plugin_reader(frd_file):
    pv.read(frd_file)

    registration = next(r for r in pv.registered_readers() if r.extension == '.frd')
    assert registration.source == 'pyvista_frd:read'
    assert registration.handler is pyvista_frd.read
    assert not registration.reader_class
    assert not registration.override


def test_extension_is_listed_as_supported():
    assert '.frd' in _reg_mod._list_custom_exts()


def test_frd_reader_is_gone():
    assert not hasattr(pv, 'FRDReader')


def test_get_reader_points_at_the_package_reader(frd_file):
    match = re.escape('use `pyvista_frd.FRDReader` directly')
    with pytest.raises(ValueError, match=match):
        pv.get_reader(frd_file)


@pytest.mark.usefixtures('frd_not_installed')
def test_read_without_the_package(frd_file):
    with pytest.raises(ImportError) as excinfo:
        pv.read(frd_file)

    message = str(excinfo.value)
    assert 'model.frd' in message
    assert 'CalculiX FRD result files (.frd)' in message
    assert '`pyvista-frd-reader` package, which is not installed' in message
    assert 'pip install pyvista[io]' in message
    assert 'pip install pyvista-frd-reader' in message


@pytest.mark.usefixtures('frd_not_installed')
def test_get_reader_without_the_package(frd_file):
    with pytest.raises(ImportError, match=re.escape('pip install pyvista[io]')):
        pv.get_reader(frd_file)


@pytest.mark.usefixtures('frd_install_broken')
def test_read_with_a_broken_install(frd_file):
    with pytest.raises(ImportError) as excinfo:
        pv.read(frd_file)

    message = str(excinfo.value)
    assert 'installed but failed to import' in message
    assert 'libpvfrd.so' in message
    # An install command cannot fix a package already present.
    assert 'pip install' not in message


@pytest.mark.usefixtures('frd_not_installed')
def test_extension_is_not_listed_without_the_package():
    assert '.frd' not in _reg_mod._list_custom_exts()


@pytest.mark.usefixtures('frd_not_installed')
def test_other_extensions_are_unaffected():
    assert _reg_mod._missing_reader_message('.vtp') is None
    assert _reg_mod._missing_reader_message('.unknown') is None


def test_entry_point_wins(frd_file):
    _reg_mod._entry_points_loaded = False
    sentinel = pv.PolyData()
    plugin = MagicMock()
    plugin.name = '.frd'
    plugin.value = 'other_package:read'
    plugin.load.return_value = lambda _path, **_kwargs: sentinel

    with patch.object(_reg_mod, 'entry_points', return_value=[plugin]):
        assert pv.read(frd_file) is sentinel

    registration = next(r for r in pv.registered_readers() if r.extension == '.frd')
    assert registration.source == 'other_package:read'


def test_explicit_registration_wins(frd_file):
    sentinel = pv.PolyData()

    pv.register_reader('.frd', lambda _path, **_kwargs: sentinel)

    assert pv.read(frd_file) is sentinel
