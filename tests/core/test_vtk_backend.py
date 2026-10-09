"""Tests for VTK backend selection and resolution in :mod:`pyvista._vtk`."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista._vtk import _resolve_root_is_flat
from pyvista._vtk import _resolve_vtk_root


def _patch_cvista_installed(monkeypatch, *, installed):
    """Make ``importlib.util.find_spec('cvista')`` report cvista as (un)installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'cvista':
            return object() if installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)


@pytest.mark.parametrize('backend', ['vtkmodules', 'cvista'])
def test_resolve_root_env_var_wins(monkeypatch, backend):
    # The env var is honored even when it contradicts what is installed.
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', backend)
    _patch_cvista_installed(monkeypatch, installed=backend != 'cvista')
    assert _resolve_vtk_root() == backend


def test_resolve_root_prefers_cvista_when_installed(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_cvista_installed(monkeypatch, installed=True)
    assert _resolve_vtk_root() == 'cvista'


def test_resolve_root_falls_back_to_vtkmodules(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_cvista_installed(monkeypatch, installed=False)
    assert _resolve_vtk_root() == 'vtkmodules'


def test_private_names_are_not_routed_to_the_backend():
    """A private lookup must not produce the developer-facing mapping message.

    Interpreter/library probes (copy, pickle, IPython) ask for unmapped dunders;
    they must not be told to add a ``module:__wrapped__`` entry to ``_vtk``.
    """
    with pytest.raises(AttributeError) as excinfo:
        _vtk.__getattr__('__wrapped__')

    assert "not defined in PyVista's vtk namespace" not in str(excinfo.value)


def test_unmapped_name_raises_attribute_error():
    """PyVista curates the VTK names it re-exports, on either backend."""
    with pytest.raises(AttributeError, match="not defined in PyVista's vtk namespace"):
        _vtk.vtkNotARealVTKClass  # noqa: B018


def test_mapped_name_resolves():
    """A mapped class resolves on whichever backend is active."""
    assert _vtk.vtkPolyData.__name__ == 'vtkPolyData'


def test_vtk_backend_reports_the_active_build():
    """``vtk_backend()`` names the build, using 'vtk' for stock VTK."""
    backend = pv.vtk_backend()
    assert isinstance(backend, str)
    if _vtk._VTK_ROOT == 'vtkmodules':
        # Reported as the familiar distribution name, not the package name.
        assert backend == 'vtk'
    else:
        assert backend == _vtk._VTK_ROOT


# Mapped classes cvista hosts in a different module than stock (``name: stock
# module``); name-based resolution makes the move invisible, needing no relocation table.
_RELOCATED_IN_CVISTA = {
    'vtkPolyDataSilhouette': 'vtkFiltersHybrid',
    'vtkGLTFReader': 'vtkIOGeometry',
}


@pytest.mark.skipif(not _vtk._VTK_ROOT_IS_FLAT, reason='requires the flat-namespace backend')
@pytest.mark.parametrize(('name', 'stock_module'), _RELOCATED_IN_CVISTA.items())
def test_flat_backend_resolves_relocated_classes(name, stock_module):
    """A relocated class resolves, and does so without using its stock module."""
    assert getattr(_vtk, name).__name__ == name
    # The stock module is still recorded (it drives the stock backend) but was not
    # used to resolve here -- that's what makes the relocation a non-event.
    assert _vtk._VTK_CLASS_TO_MODULE[name] == stock_module


@pytest.mark.skipif(_vtk._VTK_ROOT_IS_FLAT, reason='stock vtkmodules only')
def test_stock_backend_resolves_via_module_map():
    """On stock VTK the class->module mapping is still the resolution path."""
    assert _vtk._VTK_CLASS_TO_MODULE['vtkPolyData'] == 'vtkCommonDataModel'
    assert _vtk.vtkPolyData.__module__.startswith('vtkmodules')


@pytest.fixture
def fake_flat_backend(monkeypatch):
    """Point ``_vtk`` at a stand-in flat-namespace backend.

    Lets the flat resolution path be exercised on any VTK build, including the
    stock one, so the behaviour is covered by the ordinary test matrix rather
    than only when cvista happens to be installed.
    """
    root = ModuleType('_fake_flat_backend')
    root.vtkPolyData = type('vtkPolyData', (), {})
    monkeypatch.setitem(sys.modules, '_fake_flat_backend', root)
    monkeypatch.setattr(_vtk, '_VTK_ROOT', '_fake_flat_backend')
    monkeypatch.setattr(_vtk, '_VTK_ROOT_IS_FLAT', True)
    # __getattr__ caches into module globals; snapshot VALUES (a name cached
    # earlier must be restored, not just newly-added keys deleted).
    cached = dict(vars(_vtk))
    yield root
    current = vars(_vtk)
    for name in set(current) - set(cached):
        delattr(_vtk, name)
    for name, value in cached.items():
        if current.get(name) is not value:
            setattr(_vtk, name, value)


def test_flat_resolution_uses_the_root_not_the_module_map(fake_flat_backend):
    """A flat backend resolves by name off the root, ignoring the stock module."""
    assert _vtk.__getattr__('vtkPolyData') is fake_flat_backend.vtkPolyData


def test_flat_resolution_reports_a_missing_class(fake_flat_backend):  # noqa: ARG001
    """A mapped name the flat backend does not provide raises ImportError."""
    with pytest.raises(ImportError, match='Cannot import name'):
        _vtk.__getattr__('vtkSphereSource')


def test_flat_resolution_still_curates_the_namespace(fake_flat_backend):  # noqa: ARG001
    """An unmapped name is an AttributeError on a flat backend too."""
    with pytest.raises(AttributeError, match="not defined in PyVista's vtk namespace"):
        _vtk.__getattr__('vtkNotARealVTKClass')


def test_stock_root_is_not_flat():
    """``vtkmodules`` is a stock-layout package, never a flat namespace."""
    assert _resolve_root_is_flat('vtkmodules') is False


def test_flat_root_is_detected_by_probing(monkeypatch):
    """A root carrying classes directly is flat, whatever it is called."""
    root = ModuleType('_probe_flat_backend')
    root.vtkPolyData = type('vtkPolyData', (), {})
    monkeypatch.setitem(sys.modules, '_probe_flat_backend', root)

    assert _resolve_root_is_flat('_probe_flat_backend') is True


def test_stock_layout_custom_root_is_not_flat(monkeypatch):
    """A custom build laid out like stock VTK must not be treated as flat.

    Inferring this from the name alone sent every class lookup to a package root
    with no classes on it, and the failure blamed the class rather than the
    layout.
    """
    root = ModuleType('_probe_stock_layout_backend')  # no classes on the root
    monkeypatch.setitem(sys.modules, '_probe_stock_layout_backend', root)

    assert _resolve_root_is_flat('_probe_stock_layout_backend') is False


def test_missing_backend_names_the_setting():
    """An unimportable backend is reported as such, not as a missing class."""
    with pytest.raises(ImportError, match='PYVISTA_VTK_BACKEND'):
        _resolve_root_is_flat('_no_such_vtk_build')


def test_special_loaders_resolve_through_the_active_backend(fake_flat_backend):
    """No ``_SPECIAL_LOADERS`` entry may hardcode a VTK build.

    A loader importing ``vtkmodules`` directly mixes stock and backend wrapped
    types, failing later at the C++ boundary. Run against the stand-in flat backend
    so this is caught per-PR rather than only in the nightly cvista job.
    """
    for name in _vtk._SPECIAL_LOADERS:
        setattr(fake_flat_backend, name, type(name, (), {}))

    offenders = {}
    for name in _vtk._SPECIAL_LOADERS:
        resolved = _vtk.__getattr__(name)
        if resolved is not getattr(fake_flat_backend, name):
            offenders[name] = getattr(resolved, '__module__', '?')

    assert not offenders, (
        f'special loaders bypassed the active backend: {offenders}. '
        f'Resolve them with _import_from(), which routes through _VTK_ROOT.'
    )


def test_module_prefixes_match_the_active_backend():
    """``_VTK_MODULE_PREFIXES`` must match what this build's classes report.

    The override guard and ``is_vtk_attribute`` key off it and fail OPEN on a
    mismatch; this is their only coverage on the cvista backend.
    """
    from pyvista.core._vtk_utilities import _VTK_MODULE_PREFIXES

    assert _vtk.vtkPolyData.__module__.startswith(_VTK_MODULE_PREFIXES)
    assert len(_VTK_MODULE_PREFIXES) == len(set(_VTK_MODULE_PREFIXES))
