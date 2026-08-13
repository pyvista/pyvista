"""Tests for ``Plotter.export_html`` / ``export_vtksz`` deprecation shims.

The trame integration moved to :mod:`trame_pyvista` in PyVista 0.49,
so these methods now emit a deprecation warning and proxy to
``plotter.trame``. ``_trame_component`` returns that registered
component or raises :class:`ImportError` if the plugin is missing.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting.plotter import BasePlotter

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('trame_pyvista') is None,
    reason='trame_pyvista not installed',
)


@pytest.fixture
def plotter():
    pl = pv.Plotter()
    yield pl
    pl.close()


def test_trame_component_returns_registered_component(plotter):
    component = plotter._trame_component()
    assert component is plotter.trame


def test_trame_component_raises_when_unregistered():
    """``_trame_component`` raises a clear ImportError when no component is registered.

    This is what users see when ``trame-pyvista`` is not installed and
    therefore has not registered itself with :func:`register_plotter_component`.
    Use a stand-in object rather than a real Plotter so unbinding the
    descriptor does not leak into other tests.
    """

    class _StandIn:
        trame = None
        _trame_component = BasePlotter._trame_component

    with pytest.raises(ImportError, match=r'trame.*plotter component is not registered'):
        _StandIn()._trame_component()


def test_export_html_warns_and_delegates(plotter, tmp_path):
    target = tmp_path / 'scene.html'
    with pytest.warns(PyVistaDeprecationWarning, match='Plotter.export_html is deprecated'):
        result = plotter.export_html(target)
    assert result is None
    assert target.exists()
    assert target.stat().st_size > 0


def test_export_vtksz_warns_and_delegates(plotter, tmp_path):
    target = tmp_path / 'scene.vtksz'
    with pytest.warns(PyVistaDeprecationWarning, match='Plotter.export_vtksz is deprecated'):
        plotter.export_vtksz(filename=str(target))
    assert target.exists()
    assert target.stat().st_size > 0


def test_export_vtksz_returns_bytes_when_no_filename(plotter):
    with pytest.warns(PyVistaDeprecationWarning, match='Plotter.export_vtksz is deprecated'):
        data = plotter.export_vtksz(filename=None)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0


def test_trame_component_rejects_a_mismatched_vtk_build(plotter, monkeypatch):
    """A trame resolved against a different VTK build is refused, with the fix.

    A process must use one VTK build: objects cannot cross between two of them.
    Without this check the mismatch surfaces deep inside trame as an opaque
    wrapped-type ``TypeError`` instead of something the user can act on.
    """
    monkeypatch.delitem(sys.modules, 'vtk_module', raising=False)
    monkeypatch.setenv('VTK_MODULE_NAME', 'a_different_vtk_build')

    with pytest.raises(RuntimeError, match='VTK_MODULE_NAME') as excinfo:
        plotter._trame_component()

    # `show()` captures scenes under a guard that ignores a missing trame, so a
    # misconfigured one has to raise something that guard does not catch,
    # otherwise the user gets silence instead of the problem.
    assert not isinstance(excinfo.value, ImportError)
