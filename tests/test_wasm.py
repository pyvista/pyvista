"""Tests for WASM/Pyodide support module.

This file contains both unit tests (using mocks) and Pyodide integration tests.
- Unit tests: Run in standard Python environment with mocks
- Pyodide tests: Run in actual Pyodide/WebAssembly environment (marked with 'pyodide')

Run only unit tests:
    pytest tests/test_wasm.py -m "not pyodide"

Run only Pyodide tests:
    pytest tests/test_wasm.py -m pyodide --runtime=node

Run all tests:
    pytest tests/test_wasm.py
"""

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


# =============================================================================
# Pyodide Integration Tests
# =============================================================================
# These tests run in an actual Pyodide/WebAssembly environment using pytest-pyodide.
# Mark with 'pyodide' to allow selective execution.

pyodide_only = pytest.mark.pyodide


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_is_pyodide_detection(selenium):
    """Test that is_pyodide returns True in actual Pyodide environment."""
    selenium.run_js("""
        await micropip.install("pyvista");

        import pyvista as pv
        from pyvista import wasm

        # In Pyodide, is_pyodide should return True
        assert wasm.is_pyodide() is True, "is_pyodide() should return True in Pyodide"
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_plotter_creation(selenium):
    """Test WASMPlotter creation in actual Pyodide environment."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista import wasm

        # Create WASMPlotter
        plotter = wasm.WASMPlotter()
        assert plotter is not None

        # Test that internal plotter is not created until needed
        assert plotter._wasm_plotter is None
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_add_mesh(selenium):
    """Test adding mesh to WASMPlotter in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista import wasm

        # Create mesh and plotter
        mesh = pv.Sphere()
        plotter = wasm.WASMPlotter()

        # Add mesh (should not raise)
        actor = plotter.add_mesh(mesh)

        # Verify mesh was tracked
        assert len(plotter._meshes) == 1
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_generate_standalone_html(selenium):
    """Test generating standalone HTML in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista import wasm

        # Create plotter with mesh
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())

        # Generate HTML
        html = wasm.generate_standalone_html(plotter)

        # Verify HTML is generated
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html or '<html' in html.lower()
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_jupyter_backend_wasm(selenium):
    """Test setting Jupyter backend to wasm in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv

        # Set backend to wasm
        pv.set_jupyter_backend('wasm')

        # Verify backend is set
        assert pv.global_theme.jupyter.backend == 'wasm'
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_backend_auto_detection(selenium):
    """Test that wasm backend is auto-detected in actual Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista.jupyter import _resolve_backend

        # In Pyodide, should resolve to 'wasm' when pyvista-wasm is available
        backend = _resolve_backend()
        assert backend == 'wasm', f"Expected 'wasm' but got '{backend}'"
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_plotter_views(selenium):
    """Test WASMPlotter view methods in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista import wasm

        # Create plotter and add mesh
        plotter = wasm.WASMPlotter()
        plotter.add_mesh(pv.Sphere())

        # Test view methods (should not raise)
        plotter.view_xy()
        plotter.view_xz()
        plotter.view_yz()
        plotter.view_isometric()
    """)


@pytest.mark.driver_timeout(120)
@pyodide_only
def test_pyodide_wasm_plotter_background_color(selenium):
    """Test WASMPlotter background color property in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista import wasm

        # Create plotter
        plotter = wasm.WASMPlotter()

        # Set background color
        plotter.background_color = (0.1, 0.2, 0.3)

        # Get background color
        bg = plotter.background_color
        assert len(bg) == 3
    """)
