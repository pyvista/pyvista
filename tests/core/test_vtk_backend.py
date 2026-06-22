"""Tests for VTK backend selection in :mod:`pyvista._vtk`."""

from __future__ import annotations

import importlib.util

import pytest

from pyvista._vtk import _resolve_vtk_backend


def _patch_fvtk_installed(monkeypatch, *, installed):
    """Make ``importlib.util.find_spec('fvtk')`` report fvtk as (un)installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'fvtk':
            return object() if installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)


@pytest.mark.parametrize('backend', ['vtkmodules', 'fvtk'])
def test_resolve_backend_env_var_wins(monkeypatch, backend):
    # The env var is honored even when it contradicts what is installed.
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', backend)
    _patch_fvtk_installed(monkeypatch, installed=backend != 'fvtk')
    assert _resolve_vtk_backend() == backend


def test_resolve_backend_prefers_fvtk_when_installed(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_fvtk_installed(monkeypatch, installed=True)
    assert _resolve_vtk_backend() == 'fvtk'


def test_resolve_backend_falls_back_to_vtkmodules(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    _patch_fvtk_installed(monkeypatch, installed=False)
    assert _resolve_vtk_backend() == 'vtkmodules'
