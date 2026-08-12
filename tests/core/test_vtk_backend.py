"""Tests for VTK backend selection and resolution in :mod:`pyvista._vtk`."""

from __future__ import annotations

import importlib.util

import pytest

import pyvista as pv
from pyvista import _vtk
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
    """A missing private helper is an AttributeError, never a backend ImportError.

    ``_vtk.__getattr__`` must not forward private lookups to the VTK root: they
    are this module's own helpers (and interpreter probes), so forwarding turns a
    plain missing attribute into a confusing ImportError raised by the backend.
    """
    with pytest.raises(AttributeError):
        _vtk._definitely_not_a_real_helper  # noqa: B018


def test_unmapped_name_raises_attribute_error():
    """PyVista curates the VTK names it re-exports, on either backend."""
    with pytest.raises(AttributeError, match="not defined in PyVista's vtk namespace"):
        _vtk.vtkNotARealVTKClass  # noqa: B018


def test_mapped_name_resolves():
    """A mapped class resolves on whichever backend is active."""
    assert _vtk.vtkPolyData.__name__ == 'vtkPolyData'


def test_vtk_backend_reports_the_active_build():
    """``vtk_backend()`` names the build, using 'vtk' for stock VTK.

    This is the public escape hatch: it lets user code branch on -- or raise a
    clear error for -- a feature the active build does not provide, without
    reaching into ``_vtk`` internals.
    """
    backend = pv.vtk_backend()
    assert isinstance(backend, str)
    if _vtk._VTK_ROOT == 'vtkmodules':
        # Reported as the familiar distribution name, not the package name.
        assert backend == 'vtk'
    else:
        assert backend == _vtk._VTK_ROOT


# Classes PyVista maps whose hosting module differs between stock VTK and cvista
# (cvista relocates them so its core wheel tier stays rendering-free), as
# ``name: stock module``. Resolving by NAME off the package root is what makes the
# move invisible, so PyVista carries no relocation table for them.
_RELOCATED_IN_CVISTA = {
    'vtkPolyDataSilhouette': 'vtkFiltersHybrid',
    'vtkGLTFReader': 'vtkIOGeometry',
}


@pytest.mark.skipif(not _vtk._VTK_ROOT_IS_FLAT, reason='requires the flat-namespace backend')
@pytest.mark.parametrize(('name', 'stock_module'), _RELOCATED_IN_CVISTA.items())
def test_flat_backend_resolves_relocated_classes(name, stock_module):
    """A relocated class resolves, and does so without using its stock module."""
    assert getattr(_vtk, name).__name__ == name
    # PyVista still records the STOCK module (that mapping drives the stock
    # backend), yet resolution on cvista did not go through it -- which is exactly
    # what makes the relocation a non-event here.
    assert _vtk._VTK_CLASS_TO_MODULE[name] == stock_module


@pytest.mark.skipif(_vtk._VTK_ROOT_IS_FLAT, reason='stock vtkmodules only')
def test_stock_backend_resolves_via_module_map():
    """On stock VTK the class->module mapping is still the resolution path."""
    assert _vtk._VTK_CLASS_TO_MODULE['vtkPolyData'] == 'vtkCommonDataModel'
    assert _vtk.vtkPolyData.__module__.startswith('vtkmodules')
