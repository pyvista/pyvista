"""Pyodide integration tests for WASM support.

These tests run in an actual Pyodide/WebAssembly environment using pytest-pyodide.
They verify that pyvista works correctly in browser-based Python environments.
"""

from __future__ import annotations

import pytest

# Mark all tests in this file as pyodide tests
pytestmark = [
    pytest.mark.driver_timeout(120),
    pytest.mark.skip(reason='Pyodide tests require pyodide build of pyvista'),
]


def test_is_pyodide_detection(selenium):
    """Test that is_pyodide returns True in Pyodide environment."""
    selenium.run_js("""
        await micropip.install("pyvista");

        import pyvista as pv
        from pyvista import wasm

        # In Pyodide, is_pyodide should return True
        assert wasm.is_pyodide() is True, "is_pyodide() should return True in Pyodide"
    """)


def test_wasm_plotter_creation(selenium):
    """Test WASMPlotter creation in Pyodide."""
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


def test_wasm_add_mesh(selenium):
    """Test adding mesh to WASMPlotter."""
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


def test_wasm_generate_standalone_html(selenium):
    """Test generating standalone HTML."""
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


def test_jupyter_backend_wasm(selenium):
    """Test setting Jupyter backend to wasm."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv

        # Set backend to wasm
        pv.set_jupyter_backend('wasm')

        # Verify backend is set
        assert pv.global_theme.jupyter.backend == 'wasm'
    """)


def test_wasm_backend_auto_detection(selenium):
    """Test that wasm backend is auto-detected in Pyodide."""
    selenium.run_js("""
        await micropip.install("pyvista");
        await micropip.install("pyvista-wasm");

        import pyvista as pv
        from pyvista.jupyter import _resolve_backend

        # In Pyodide, should resolve to 'wasm' when pyvista-wasm is available
        backend = _resolve_backend()
        assert backend == 'wasm', f"Expected 'wasm' but got '{backend}'"
    """)


def test_wasm_plotter_views(selenium):
    """Test WASMPlotter view methods."""
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


def test_wasm_plotter_background_color(selenium):
    """Test WASMPlotter background color property."""
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
