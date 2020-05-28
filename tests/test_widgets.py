import numpy as np
import pytest

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
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda box, widget: box # Does nothing
    p.add_mesh(mesh)
    p.add_box_widget(callback=func, pass_widget=True)
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
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin, widget: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, implicit=False)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda normal, origin, widget: normal # Does nothing
    p.add_mesh(mesh)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=False)
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
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda line, widget: line # Does nothing
    p.add_mesh(mesh)
    p.add_line_widget(callback=func, pass_widget=True)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda a, b: (a, b) # Does nothing
    p.add_mesh(mesh)
    p.add_line_widget(callback=func, use_vertices=True)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_text_slider():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value: value # Does nothing
    p.add_mesh(mesh)
    with pytest.raises(TypeError, match='must be a list'):
        p.add_text_slider_widget(callback=func, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        p.add_text_slider_widget(callback=func, data=[])
    p.add_text_slider_widget(callback=func, data=['foo', 'bar'])
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_slider():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value: value # Does nothing
    p.add_mesh(mesh)
    p.add_slider_widget(callback=func, rng=[0,10], style="classic")
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    for event_type in ['start', 'end', 'always']:
        p.add_slider_widget(callback=func, rng=[0,10],
                            event_type=event_type)
    with pytest.raises(TypeError, match='type for ``style``'):
        p.add_slider_widget(callback=func, rng=[0,10], style=0)
    with pytest.raises(KeyError, match='styles available'):
        p.add_slider_widget(callback=func, rng=[0,10], style="foo")
    with pytest.raises(TypeError, match='type for `event_type`'):
        p.add_slider_widget(callback=func, rng=[0,10],
                            event_type=0)
    with pytest.raises(ValueError, match='value for `event_type`'):
        p.add_slider_widget(callback=func, rng=[0,10],
                            event_type='foo')
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value, widget: value # Does nothing
    p.add_mesh(mesh)
    p.add_slider_widget(callback=func, rng=[0,10], style="modern",
                        pass_widget=True)
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
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda spline, widget: spline # Does nothing
    p.add_mesh(mesh)
    p.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    p.close()

    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    p.add_mesh_slice_spline(mesh)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_sphere():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda center: center # Does nothing
    p.add_sphere_widget(callback=func, center=(0, 0, 0))
    p.close()

    nodes = np.array([[-1,-1,-1], [1,1,1]])
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda center: center # Does nothing
    p.add_sphere_widget(callback=func, center=nodes)
    p.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_widget_checkbox_button():
    p = pyvista.Plotter(off_screen=OFF_SCREEN)
    func = lambda value: value # Does nothing
    p.add_mesh(mesh)
    p.add_checkbox_button_widget(callback=func)
    p.close()
