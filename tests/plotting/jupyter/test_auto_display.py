"""Test automatic display of Plotter in Jupyter notebooks via _repr_html_."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

import pyvista as pv

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

has_ipython = bool(importlib.util.find_spec('IPython'))

skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


class TestPlotterReprHTML:
    """Test the _repr_html_ method of Plotter class."""

    def test_repr_html_not_in_jupyter(self):
        """Test that _repr_html_ returns None when not in Jupyter."""
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())
        
        # Should return None when not in Jupyter
        assert plotter._repr_html_() is None

    @skip_no_ipython
    def test_repr_html_with_ipython_none(self, mocker: MockerFixture):
        """Test that _repr_html_ returns None when IPython.get_ipython() is None."""
        # Mock IPython to return None
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = None
        
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())
        
        assert plotter._repr_html_() is None

    @skip_no_ipython
    def test_repr_html_backend_none(self, mocker: MockerFixture):
        """Test that _repr_html_ returns None when jupyter_backend is 'none'."""
        # Mock IPython to return a non-None value
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = object()  # Any non-None value
        
        # Set backend to 'none'
        original_backend = pv.global_theme.jupyter_backend
        try:
            pv.set_jupyter_backend('none')
            
            plotter = pv.Plotter()
            plotter.add_mesh(pv.Sphere())
            
            assert plotter._repr_html_() is None
        finally:
            pv.set_jupyter_backend(original_backend)

    @skip_no_ipython
    @pytest.mark.skip_plotting
    def test_repr_html_static_backend(self, mocker: MockerFixture):
        """Test that _repr_html_ returns HTML for static backend."""
        # Mock IPython to return a non-None value
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = object()  # Any non-None value
        
        # Set backend to 'static'
        original_backend = pv.global_theme.jupyter_backend
        try:
            pv.set_jupyter_backend('static')
            
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())
            
            html = plotter._repr_html_()
            
            # Should return HTML with base64 encoded image
            assert html is not None
            assert '<img src="data:image/png;base64,' in html
            assert html.endswith('" />')
        finally:
            pv.set_jupyter_backend(original_backend)

    @skip_no_ipython
    def test_repr_html_handles_exceptions(self, mocker: MockerFixture):
        """Test that _repr_html_ handles exceptions gracefully."""
        # Mock IPython to return a non-None value
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = object()  # Any non-None value
        
        # Mock handle_plotter to raise an exception
        handle_mock = mocker.patch('pyvista.jupyter.notebook.handle_plotter')
        handle_mock.side_effect = Exception("Test exception")
        
        plotter = pv.Plotter()
        plotter.add_mesh(pv.Sphere())
        
        # Should return None and not raise
        assert plotter._repr_html_() is None

    @skip_no_ipython
    @pytest.mark.skip_plotting
    def test_repr_html_with_trame_backend(self, mocker: MockerFixture):
        """Test that _repr_html_ works with trame backend (when available)."""
        # Check if trame is available
        try:
            import trame  # noqa: F401
            has_trame = True
        except ImportError:
            has_trame = False
        
        if not has_trame:
            pytest.skip("Requires trame package")
        
        # Mock IPython to return a non-None value
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = object()  # Any non-None value
        
        # Set backend to 'trame'
        original_backend = pv.global_theme.jupyter_backend
        try:
            pv.set_jupyter_backend('trame')
            
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv.Sphere())
            
            html = plotter._repr_html_()
            
            # Should return HTML (either from widget or fallback to static)
            assert html is not None
        finally:
            pv.set_jupyter_backend(original_backend)

    @skip_no_ipython
    @pytest.mark.skip_plotting
    def test_repr_html_integration(self, mocker: MockerFixture):
        """Integration test: verify _repr_html_ produces valid HTML."""
        # Mock IPython to return a non-None value
        ipython_mock = mocker.patch('pyvista.plotting.plotter.get_ipython')
        ipython_mock.return_value = object()  # Any non-None value
        
        # Use static backend for predictable output
        original_backend = pv.global_theme.jupyter_backend
        try:
            pv.set_jupyter_backend('static')
            
            # Create a plotter with some content
            plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
            plotter.add_mesh(pv.Cube(), color='red')
            plotter.add_mesh(pv.Sphere(center=(2, 0, 0)), color='blue')
            
            html = plotter._repr_html_()
            
            # Verify HTML structure
            assert html is not None
            assert html.startswith('<img src="data:image/png;base64,')
            assert html.endswith('" />')
            
            # Extract base64 data
            start = html.find('base64,') + 7
            end = html.find('"', start)
            base64_data = html[start:end]
            
            # Verify it's valid base64
            import base64
            try:
                decoded = base64.b64decode(base64_data)
                assert len(decoded) > 0  # Should have some image data
            except Exception:
                pytest.fail("Invalid base64 data in HTML output")
                
        finally:
            pv.set_jupyter_backend(original_backend)