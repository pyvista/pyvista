"""Tests for formats served by companion packages rather than by PyVista."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import numpy as np
import pytest
import pyvista_frd
import pyvista_zstd

import pyvista as pv
from pyvista.core.utilities import reader_registry as _reg_mod
from pyvista.core.utilities import writer_registry as _writer_mod

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
    """Undo the lazy registration an optional-format lookup performs."""
    reader_state = _reg_mod._save_registry_state()
    writer_state = _writer_mod._save_registry_state()
    yield
    _reg_mod._restore_registry_state(reader_state)
    _writer_mod._restore_registry_state(writer_state)


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
    """Report ``module`` as present, then fail to execute it."""

    def __init__(self, module, message):
        self.module = module
        self.message = message

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        return importlib.machinery.ModuleSpec(fullname, self) if fullname == self.module else None

    def exec_module(self, module):  # noqa: ARG002
        raise ImportError(self.message)


def _break_install(monkeypatch, module, message):
    """Leave ``module`` importable-looking but failing to execute."""
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenFinder(module, message), *sys.meta_path])


@pytest.fixture
def frd_install_broken(monkeypatch):
    _break_install(monkeypatch, 'pyvista_frd', 'libpvfrd.so: cannot open shared object file')


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


ZSTD_EXTENSIONS = ['.pv', '.zvtk']


def _forget_zstd():
    """Drop every trace of ``pyvista_zstd`` from both registries."""
    for ext in ZSTD_EXTENSIONS:
        _reg_mod._custom_ext_readers.pop(ext, None)
        _reg_mod._custom_class_readers.pop(ext, None)
        _reg_mod._custom_ext_reader_sources.pop(ext, None)
        _reg_mod._pending_ext_readers.pop(ext, None)
        _writer_mod._custom_ext_writers.pop(ext, None)
        _writer_mod._custom_ext_writer_sources.pop(ext, None)
        _writer_mod._pending_ext_writers.pop(ext, None)


@pytest.fixture
def zstd_file(tmp_path):
    path = tmp_path / 'sphere.pv'
    pyvista_zstd.write(pv.Sphere(), path)
    return str(path)


def _hide_zstd_entry_points(monkeypatch):
    """Hide the entry points ``pyvista-zstd`` declares for its two extensions."""
    monkeypatch.setattr(_reg_mod, 'entry_points', lambda **_kwargs: [])
    monkeypatch.setattr(_writer_mod, 'entry_points', lambda **_kwargs: [])
    _forget_zstd()


def _uninstall_zstd(monkeypatch):
    """Present what a stock PyVista install sees: no module, no entry points."""
    _hide_zstd_entry_points(monkeypatch)
    monkeypatch.setitem(sys.modules, 'pyvista_zstd', None)


@pytest.fixture
def zstd_not_installed(monkeypatch):
    _uninstall_zstd(monkeypatch)


@pytest.fixture
def zstd_install_broken(monkeypatch):
    _hide_zstd_entry_points(monkeypatch)
    _break_install(monkeypatch, 'pyvista_zstd', 'libzstd.so: cannot open shared object file')


def test_zstd_get_reader_points_at_the_package_reader(zstd_file):
    match = re.escape('use `pyvista_zstd.Reader` directly')
    with pytest.raises(ValueError, match=match):
        pv.get_reader(zstd_file)


def test_zvtk_still_round_trips_and_says_it_is_legacy(tmp_path):
    mesh = pv.Sphere()
    path = tmp_path / 'sphere.zvtk'

    with pytest.warns(FutureWarning, match='legacy zvtk'):
        mesh.save(path)

    assert pv.read(path) == mesh


@pytest.mark.parametrize('ext', ZSTD_EXTENSIONS)
def test_zstd_read_without_the_package(tmp_path, monkeypatch, ext):
    """Save a file with the package installed, then read it without."""
    path = tmp_path / f'sphere{ext}'
    with warnings.catch_warnings():
        # ``.zvtk`` is legacy; test_zvtk_still_round_trips_and_says_it_is_legacy covers that.
        warnings.simplefilter('ignore', FutureWarning)
        pyvista_zstd.write(pv.Sphere(), path)

    _uninstall_zstd(monkeypatch)

    with pytest.raises(ImportError) as excinfo:
        pv.read(path)

    message = str(excinfo.value)
    assert f'sphere{ext}' in message
    assert f'({ext}) requires the `pyvista-zstd` package, which is not installed' in message
    assert 'pip install pyvista[io]' in message
    assert 'pip install pyvista-zstd' in message


@pytest.mark.usefixtures('zstd_not_installed')
def test_zstd_save_without_the_package(tmp_path):
    path = tmp_path / 'sphere.pv'

    with pytest.raises(ImportError) as excinfo:
        pv.Sphere().save(path)

    message = str(excinfo.value)
    assert 'sphere.pv' in message
    assert "Writing PyVista's native zstd-compressed format (.pv)" in message
    assert '`pyvista-zstd` package, which is not installed' in message
    assert 'pip install pyvista[io]' in message
    assert 'pip install pyvista-zstd' in message
    assert not path.exists()


@pytest.mark.usefixtures('zstd_not_installed')
def test_zvtk_is_offered_for_reading_but_not_for_writing(tmp_path):
    assert _reg_mod._missing_reader_message('.zvtk') is not None

    with pytest.raises(ValueError, match='Invalid file extension') as excinfo:
        pv.Sphere().save(tmp_path / 'sphere.zvtk')

    assert 'pyvista-zstd' not in str(excinfo.value)


@pytest.mark.usefixtures('zstd_not_installed')
def test_zstd_get_reader_without_the_package(zstd_file):
    with pytest.raises(ImportError, match=re.escape('pip install pyvista[io]')):
        pv.get_reader(zstd_file)


@pytest.mark.usefixtures('zstd_install_broken')
def test_zstd_read_with_a_broken_install(zstd_file):
    with pytest.raises(ImportError) as excinfo:
        pv.read(zstd_file)

    message = str(excinfo.value)
    assert 'installed but failed to import' in message
    assert 'libzstd.so' in message
    # An install command cannot fix a package already present.
    assert 'pip install' not in message


@pytest.mark.usefixtures('zstd_install_broken')
def test_zstd_save_with_a_broken_install(tmp_path):
    with pytest.raises(ImportError) as excinfo:
        pv.Sphere().save(tmp_path / 'sphere.pv')

    message = str(excinfo.value)
    assert 'installed but failed to import' in message
    assert 'libzstd.so' in message
    assert 'pip install' not in message


def test_writer_side_leaves_other_extensions_alone():
    assert _writer_mod._missing_writer_message('.vtp') is None
    assert _writer_mod._missing_writer_message('.unknown') is None
    # ``.frd`` is read-only, so the writer side must not claim it.
    assert _writer_mod._missing_writer_message('.frd') is None


@pytest.mark.usefixtures('zstd_not_installed')
def test_unknown_extension_still_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match='Invalid file extension'):
        pv.Sphere().save(tmp_path / 'sphere.unknown')


@pytest.mark.usefixtures('zstd_not_installed')
def test_a_registered_writer_beats_the_missing_package_error(tmp_path):
    written = []

    pv.register_writer('.pv', lambda _dataset, path, **_kwargs: written.append(path))

    path = tmp_path / 'sphere.pv'
    with pytest.raises(OSError, match='Custom writer failed to write file'):
        pv.Sphere().save(path)
    assert written == [str(path)]
