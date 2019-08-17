import pytest
import sys

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

# Widgets cannot be used off screen - they must have an interactive renderer
OFF_SCREEN = False


mesh = examples.load_uniform()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_box():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda box: box # Does nothing
    p.add_mesh(mesh)
    p.enable_box_widget(callback=func)
    p.disable_box_widget()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_clip_box(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_plane():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin: normal # Does nothing
    p.add_mesh(mesh)
    p.enable_plane_widget(callback=func)
    p.disable_plane_widget()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_clip_plane(mesh)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_slice(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_line():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda line: line # Does nothing
    p.add_mesh(mesh)
    p.enable_line_widget(callback=func)
    p.disable_line_widget()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda a, b: (a, b) # Does nothing
    p.add_mesh(mesh)
    p.enable_line_widget(callback=func, use_vertices=True)
    p.disable_line_widget()
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_slider():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value: value # Does nothing
    p.add_mesh(mesh)
    p.enable_slider_widget(callback=func, rng=[0,10])
    p.disable_slider_widget()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_threshold(mesh)
    p.add_mesh(mesh.outline())
    p.close()
