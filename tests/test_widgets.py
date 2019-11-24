import numpy as np
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
    p.add_box_widget(callback=func)
    p.clear_box_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda box, widget: box # Does nothing
    p.add_mesh(mesh)
    p.add_box_widget(callback=func, pass_widget=True)
    p.clear_box_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_clip_box(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_plane():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, implicit=True)
    p.clear_plane_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin, widget: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    p.clear_plane_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, implicit=False)
    p.clear_plane_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin, widget: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=False)
    p.clear_plane_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_clip_plane(mesh)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_slice(mesh)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_slice_orthogonal(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_line():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda line: line # Does nothing
    p.add_mesh(mesh)
    p.add_line_widget(callback=func)
    p.clear_line_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda line, widget: line # Does nothing
    p.add_mesh(mesh)
    p.add_line_widget(callback=func, pass_widget=True)
    p.clear_line_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda a, b: (a, b) # Does nothing
    p.add_mesh(mesh)
    p.add_line_widget(callback=func, use_vertices=True)
    p.clear_line_widgets()
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_slider():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value: value # Does nothing
    p.add_mesh(mesh)
    p.add_slider_widget(callback=func, rng=[0,10])
    p.clear_slider_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value, widget: value # Does nothing
    p.add_mesh(mesh)
    p.add_slider_widget(callback=func, rng=[0,10], pass_widget=True)
    p.clear_slider_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_threshold(mesh, invert=True)
    p.add_mesh(mesh.outline())
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_threshold(mesh, invert=False)
    p.add_mesh(mesh.outline())
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_isovalue(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_spline():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda spline: spline # Does nothing
    p.add_mesh(mesh)
    p.add_spline_widget(callback=func)
    p.clear_spline_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda spline, widget: spline # Does nothing
    p.add_mesh(mesh)
    p.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    p.clear_spline_widgets()
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_slice_spline(mesh)
    p.clear_spline_widgets()
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_sphere():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda center: center # Does nothing
    p.add_sphere_widget(callback=func, center=(0, 0, 0))
    p.clear_sphere_widgets()
    p.close()

    nodes = np.array([[-1,-1,-1], [1,1,1]])
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda center: center # Does nothing
    p.add_sphere_widget(callback=func, center=nodes)
    p.clear_sphere_widgets()
    p.close()
