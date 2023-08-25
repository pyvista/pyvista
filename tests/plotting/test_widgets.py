import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def test_widget_box(uniform):
    p = pv.Plotter()
    func = lambda box: box  # Does nothing
    p.add_mesh(uniform)
    p.add_box_widget(callback=func)
    p.close()

    p = pv.Plotter()
    func = lambda box, widget: box  # Does nothing
    p.add_mesh(uniform)
    p.add_box_widget(callback=func, pass_widget=True)
    p.close()

    # clip box with and without crinkle
    p = pv.Plotter()
    p.add_mesh_clip_box(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_box(uniform, crinkle=True)
    p.close()

    p = pv.Plotter()
    # merge_points=True is the default and is tested above
    p.add_mesh_clip_box(uniform, merge_points=False)
    p.close()


def test_widget_plane(uniform):
    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, implicit=False)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, pass_widget=True, implicit=False)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, assign_to_axis='z', implicit=True)
    p.close()

    p = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing
    p.add_mesh(uniform)
    p.add_plane_widget(callback=func, normal_rotation=False, implicit=False)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_plane(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_clip_plane(uniform, crinkle=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice(uniform)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice_orthogonal(uniform)
    p.close()


def test_widget_line(uniform):
    p = pv.Plotter()
    func = lambda line: line  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func)
    p.close()

    p = pv.Plotter()
    func = lambda line, widget: line  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func, pass_widget=True)
    p.close()

    p = pv.Plotter()
    func = lambda a, b: (a, b)  # Does nothing
    p.add_mesh(uniform)
    p.add_line_widget(callback=func, use_vertices=True)
    p.close()


def test_widget_text_slider(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    with pytest.raises(TypeError, match='must be a list'):
        p.add_text_slider_widget(callback=func, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        p.add_text_slider_widget(callback=func, data=[])
    for style in pv.global_theme.slider_styles:
        p.add_text_slider_widget(callback=func, data=['foo', 'bar'], style=style)
    p.close()


def test_widget_slider(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    p.add_slider_widget(callback=func, rng=[0, 10], style="classic")
    p.close()

    p = pv.Plotter()
    for interaction_event in ['start', 'end', 'always']:
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event=interaction_event)
    with pytest.raises(TypeError, match='type for ``style``'):
        p.add_slider_widget(callback=func, rng=[0, 10], style=0)
    with pytest.raises(AttributeError):
        p.add_slider_widget(callback=func, rng=[0, 10], style="foo")
    with pytest.raises(TypeError, match='Expected type for `interaction_event`'):
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event=0)
    with pytest.raises(ValueError, match='Expected value for `interaction_event`'):
        p.add_slider_widget(callback=func, rng=[0, 10], interaction_event='foo')
    p.close()

    p = pv.Plotter()
    func = lambda value, widget: value  # Does nothing
    p.add_mesh(uniform)
    p.add_slider_widget(callback=func, rng=[0, 10], style="modern", pass_widget=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_threshold(uniform, invert=True)
    p.add_mesh(uniform.outline())
    p.close()

    p = pv.Plotter()
    p.add_mesh_threshold(uniform, invert=False)
    p.add_mesh(uniform.outline())
    p.close()

    p = pv.Plotter()
    p.add_mesh_isovalue(uniform)
    p.close()

    func = lambda value: value  # Does nothing
    p = pv.Plotter()
    title_height = np.random.random()
    s = p.add_slider_widget(callback=func, rng=[0, 10], style="classic", title_height=title_height)
    assert s.GetRepresentation().GetTitleHeight() == title_height
    p.close()

    p = pv.Plotter()
    title_opacity = np.random.random()
    s = p.add_slider_widget(
        callback=func, rng=[0, 10], style="classic", title_opacity=title_opacity
    )
    assert s.GetRepresentation().GetTitleProperty().GetOpacity() == title_opacity
    p.close()

    p = pv.Plotter()
    title_color = "red"
    s = p.add_slider_widget(callback=func, rng=[0, 10], style="classic", title_color=title_color)
    assert s.GetRepresentation().GetTitleProperty().GetColor() == pv.Color(title_color)
    p.close()

    p = pv.Plotter()
    fmt = "%0.9f"
    s = p.add_slider_widget(callback=func, rng=[0, 10], style="classic", fmt=fmt)
    assert s.GetRepresentation().GetLabelFormat() == fmt
    p.close()

    # custom width
    p = pv.Plotter()
    slider = p.add_slider_widget(
        callback=func, rng=[0, 10], fmt=fmt, tube_width=0.1, slider_width=0.2
    )
    assert slider.GetRepresentation().GetSliderWidth() == 0.2
    assert slider.GetRepresentation().GetTubeWidth() == 0.1
    p.close()


def test_widget_spline(uniform):
    p = pv.Plotter()
    func = lambda spline: spline  # Does nothing
    p.add_mesh(uniform)
    p.add_spline_widget(callback=func)
    p.close()

    p = pv.Plotter()
    p.add_mesh(uniform)
    pts = np.array([[1, 5, 4], [2, 4, 9], [3, 6, 2]])
    with pytest.raises(ValueError, match='`initial_points` must be length `n_handles`'):
        p.add_spline_widget(callback=func, n_handles=4, initial_points=pts)
    p.add_spline_widget(callback=func, n_handles=3, initial_points=pts)
    p.close()

    p = pv.Plotter()
    func = lambda spline, widget: spline  # Does nothing
    p.add_mesh(uniform)
    p.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    p.close()

    p = pv.Plotter()
    p.add_mesh_slice_spline(uniform)
    p.close()


def test_measurement_widget():
    class DistanceCallback:
        def __init__(self):
            self.called = False
            self.args = None
            self.count = 0

        def __call__(self, *args, **kwargs):
            self.called = True
            self.args = args
            self.kwargs = kwargs
            self.count += 1

    p = pv.Plotter(window_size=[1000, 1000])
    p.add_mesh(examples.load_random_hills())
    distance_callback = DistanceCallback()
    p.add_measurement_widget(callback=distance_callback)
    p.view_xy()
    p.show(auto_close=False)
    width, height = p.window_size

    p.iren._mouse_left_button_click(300, 300)
    p.iren._mouse_left_button_click(700, 700)

    assert distance_callback.called
    assert pytest.approx(distance_callback.args[2], 1.0) == 17.4

    p.close()


def test_widget_sphere(uniform):
    p = pv.Plotter()
    func = lambda center: center  # Does nothing
    p.add_sphere_widget(callback=func, center=(0, 0, 0))
    p.close()

    # pass multiple centers
    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    p = pv.Plotter()
    func = lambda center, index: center  # Does nothing
    p.add_sphere_widget(callback=func, center=nodes)
    p.close()


def test_widget_checkbox_button(uniform):
    p = pv.Plotter()
    func = lambda value: value  # Does nothing
    p.add_mesh(uniform)
    p.add_checkbox_button_widget(callback=func)
    p.close()


def test_widget_closed(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_checkbox_button_widget(callback=lambda value: value)


@pytest.mark.needs_vtk_version(9, 1)
def test_add_camera_orientation_widget(uniform):
    p = pv.Plotter()
    p.add_camera_orientation_widget()
    assert p.camera_widgets
    p.close()
    assert not p.camera_widgets


def test_plot_algorithm_widgets():
    algo = vtk.vtkRTAnalyticSource()

    pl = pv.Plotter()
    pl.add_mesh_clip_box(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_isovalue(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(algo)
    pl.close()


def test_add_volume_clip_plane(uniform):
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='The `volume` parameter type must'):
        pl.add_volume_clip_plane(pv.Sphere())

    widget = pl.add_volume_clip_plane(uniform)
    assert isinstance(widget, vtk.vtkImplicitPlaneWidget)
    assert pl.volume.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()

    pl = pv.Plotter()
    vol = pl.add_volume(uniform)
    assert vol.mapper.GetClippingPlanes() is None
    pl.add_volume_clip_plane(vol)
    assert vol.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_plot_pointset_widgets(pointset):
    pointset = pointset.elevation()

    assert isinstance(pointset, pv.PointSet)

    pl = pv.Plotter()
    pl.add_mesh_clip_box(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(pointset)
    pl.close()
