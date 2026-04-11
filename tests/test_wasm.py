"""Tests for WASM/Pyodide support module."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import pyvista as pv
from pyvista import wasm


class TestIsPyodide:
    """Tests for is_pyodide function."""

    def test_is_pyodide_returns_false_in_standard_python(self):
        """Test that is_pyodide returns False in standard Python."""
        assert wasm.is_pyodide() is False

    @mock.patch.object(sys, 'platform', 'emscripten')
    def test_is_pyodide_returns_true_in_emscripten(self):
        """Test that is_pyodide returns True in emscripten platform."""
        assert wasm.is_pyodide() is True


class TestWASMBackendIntegration:
    """Tests for WASM backend integration with Jupyter."""

    def test_wasm_in_allowed_backends(self):
        """Test that 'wasm' is in the list of allowed Jupyter backends."""
        from pyvista.jupyter import ALLOWED_BACKENDS

        assert 'wasm' in ALLOWED_BACKENDS

    def test_jupyter_backend_validation_accepts_wasm(self):
        """Test that the jupyter backend validation accepts 'wasm'."""
        from pyvista.jupyter import _validate_jupyter_backend

        # Should raise ImportError since pyvista-wasm is not installed
        with pytest.raises(ImportError, match='pyvista-wasm'):
            _validate_jupyter_backend('wasm')

    def test_jupyter_backend_validation_is_case_insensitive(self):
        """Test that backend validation is case insensitive for wasm."""
        from pyvista.jupyter import _validate_jupyter_backend

        # Should raise ImportError since pyvista-wasm is not installed
        with pytest.raises(ImportError, match='pyvista-wasm'):
            _validate_jupyter_backend('WASM')


class TestWASMPlotter:
    """Tests for WASMPlotter class."""

    def test_wasm_plotter_requires_pyvista_wasm(self):
        """Test that WASMPlotter requires pyvista-wasm package."""
        wasm_plotter = wasm.WASMPlotter()
        with pytest.raises(ImportError, match='pyvista-wasm'):
            # Access the internal plotter to trigger the import
            wasm_plotter._get_wasm_plotter()

    def test_wasm_plotter_add_mesh_without_pyvista_wasm(self):
        """Test that add_mesh raises ImportError without pyvista-wasm."""
        mesh = pv.Sphere()
        wasm_plotter = wasm.WASMPlotter()

        with pytest.raises(ImportError, match='pyvista-wasm'):
            wasm_plotter.add_mesh(mesh)


class TestGenerateStandaloneHTML:
    """Tests for generate_standalone_html function."""

    def test_generate_standalone_html_requires_pyvista_wasm(self):
        """Test that generate_standalone_html requires pyvista-wasm."""
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())

        with pytest.raises(ImportError, match='pyvista-wasm'):
            wasm.generate_standalone_html(plotter)


class TestJupyterBackendAutoDetection:
    """Tests for WASM backend auto-detection in Pyodide environments."""

    @mock.patch.object(sys, 'platform', 'emscripten')
    def test_resolve_backend_prefers_wasm_in_pyodide(self):
        """Test that _resolve_backend prefers wasm in emscripten environment."""
        from pyvista.jupyter import _resolve_backend

        # Mock pyvista_wasm as available
        with mock.patch.dict('sys.modules', {'pyvista_wasm': mock.MagicMock()}):
            backend = _resolve_backend()
            assert backend == 'wasm'

    @mock.patch.object(sys, 'platform', 'emscripten')
    def test_resolve_backend_fallback_without_pyvista_wasm(self):
        """Test fallback when pyvista-wasm is not available in emscripten."""
        from pyvista.jupyter import _resolve_backend

        # Without pyvista_wasm, should fall back to trame or static
        backend = _resolve_backend()
        # Should not be wasm since pyvista_wasm is not available
        assert backend != 'wasm'
        assert backend in ['trame', 'static']


class TestNotebookWasmHandler:
    """Tests for WASM handler in notebook module."""

    def test_show_wasm_requires_pyvista_wasm(self):
        """Test that show_wasm requires pyvista-wasm package."""
        from pyvista.jupyter.notebook import show_wasm

        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())

        with pytest.raises(ImportError, match='pyvista-wasm'):
            show_wasm(plotter)


class TestWasmModuleExports:
    """Tests for wasm module exports."""

    def test_wasm_module_has_is_pyodide(self):
        """Test that wasm module exports is_pyodide function."""
        assert hasattr(wasm, 'is_pyodide')
        assert callable(wasm.is_pyodide)

    def test_wasm_module_has_wasm_plotter(self):
        """Test that wasm module exports WASMPlotter class."""
        assert hasattr(wasm, 'WASMPlotter')
        assert isinstance(wasm.WASMPlotter, type)

    def test_wasm_module_has_generate_standalone_html(self):
        """Test that wasm module exports generate_standalone_html function."""
        assert hasattr(wasm, 'generate_standalone_html')
        assert callable(wasm.generate_standalone_html)

    def test_wasm_available_from_main_namespace(self):
        """Test that wasm is accessible from the main pyvista namespace."""
        assert hasattr(pv, 'wasm')
