"""Tests for VTK backend selection in :mod:`pyvista._vtk`."""

from __future__ import annotations

import importlib.util

import pytest

from pyvista._vtk import _resolve_vtk_backend


def _patch_cvista_installed(monkeypatch, *, installed):
    """Make ``importlib.util.find_spec('cvista')`` report cvista as (un)installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'cvista':
            return object() if installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)


@pytest.mark.parametrize('backend', ['vtkmodules', 'cvista'])
def test_resolve_backend_env_var_wins(monkeypatch, backend):
    # The env var is honored even when it contradicts what is installed.
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', backend)
    _patch_cvista_installed(monkeypatch, installed=backend != 'cvista')
    assert _resolve_vtk_backend() == backend


def test_resolve_backend_prefers_cvista_when_installed(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_cvista_installed(monkeypatch, installed=True)
    assert _resolve_vtk_backend() == 'cvista'


def test_resolve_backend_falls_back_to_vtkmodules(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_cvista_installed(monkeypatch, installed=False)
    assert _resolve_vtk_backend() == 'vtkmodules'


def test_cvista_relocated_class_map_is_flattened():
    from pyvista._vtk import _CVISTA_RELOCATED
    from pyvista._vtk import _CVISTA_RELOCATED_CLASS

    expected = {
        cls: module for moved in _CVISTA_RELOCATED.values() for cls, module in moved.items()
    }
    assert expected == _CVISTA_RELOCATED_CLASS


def test_cvista_finder_grafts_relocated_class(monkeypatch):
    import importlib
    from types import ModuleType

    from pyvista._vtk import _VtkmodulesToCvistaFinder

    # A class cvista relocated out of vtkFiltersHybrid into vtkFiltersHybridRendering.
    original = ModuleType('cvista.vtkFiltersHybrid')  # no vtkPolyDataSilhouette here
    relocated = ModuleType('cvista.vtkFiltersHybridRendering')
    relocated.vtkPolyDataSilhouette = object()

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == 'cvista.vtkFiltersHybridRendering':
            return relocated
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', fake_import)
    _VtkmodulesToCvistaFinder._graft_relocated(original, 'vtkFiltersHybrid')
    assert original.vtkPolyDataSilhouette is relocated.vtkPolyDataSilhouette


def test_cvista_finder_graft_is_soft_when_target_missing(monkeypatch):
    import importlib
    from types import ModuleType

    from pyvista._vtk import _VtkmodulesToCvistaFinder

    # On a cvista build predating the move, the new module does not exist yet:
    # the graft must be a no-op, not an error.
    original = ModuleType('cvista.vtkFiltersHybrid')
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == 'cvista.vtkFiltersHybridRendering':
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', fake_import)
    _VtkmodulesToCvistaFinder._graft_relocated(original, 'vtkFiltersHybrid')
    assert not hasattr(original, 'vtkPolyDataSilhouette')
