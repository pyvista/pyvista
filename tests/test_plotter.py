"""Test any plotting that does not require rendering.

All other tests requiring rendering should to in
./plotting/test_plotting.py

"""
import pytest
import pyvista


def test_plotter_image():
    plotter = pyvista.Plotter()
    with pytest.raises(AttributeError, match='not yet been setup'):
        plotter.image
