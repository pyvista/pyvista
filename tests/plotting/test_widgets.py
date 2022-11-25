import numpy as np
import pytest

import pyvista
from pyvista.plotting import system_supports_plotting

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip


def noop_callback(*args, **kwargs):
    """A callback that doesn't do anything."""
    pass


def test_widget_box(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_box_widget(callback=noop_callback)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_box_widget(callback=noop_callback, pass_widget=True)
    p.close()

    # clip box with and without crinkle
    p = pyvista.Plotter()
    p.add_mesh_clip_box(uniform)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_clip_box(uniform, crinkle=True)
    p.close()

    p = pyvista.Plotter()
    # merge_points=True is the default and is tested above
    p.add_mesh_clip_box(uniform, merge_points=False)
    p.close()


def test_widget_plane(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, implicit=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, pass_widget=True, implicit=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, implicit=False)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, pass_widget=True, implicit=False)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, assign_to_axis='z', implicit=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_plane_widget(callback=noop_callback, normal_rotation=False, implicit=False)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_clip_plane(uniform)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_clip_plane(uniform, crinkle=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_slice(uniform)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_slice_orthogonal(uniform)
    p.close()


def test_widget_line(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_line_widget(callback=noop_callback)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_line_widget(callback=noop_callback, pass_widget=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_line_widget(callback=noop_callback, use_vertices=True)
    p.close()


def test_widget_text_slider(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    with pytest.raises(TypeError, match='must be a list'):
        p.add_text_slider_widget(callback=noop_callback, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        p.add_text_slider_widget(callback=noop_callback, data=[])
    for style in pyvista.global_theme.slider_styles:
        p.add_text_slider_widget(callback=noop_callback, data=['foo', 'bar'], style=style)
    p.close()


def test_widget_slider(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_slider_widget(callback=noop_callback, rng=[0, 10], style="classic")
    p.close()

    p = pyvista.Plotter()
    for event_type in ['start', 'end', 'always']:
        p.add_slider_widget(callback=noop_callback, rng=[0, 10], event_type=event_type)
    with pytest.raises(TypeError, match='type for ``style``'):
        p.add_slider_widget(callback=noop_callback, rng=[0, 10], style=0)
    with pytest.raises(AttributeError):
        p.add_slider_widget(callback=noop_callback, rng=[0, 10], style="foo")
    with pytest.raises(TypeError, match='type for `event_type`'):
        p.add_slider_widget(callback=noop_callback, rng=[0, 10], event_type=0)
    with pytest.raises(ValueError, match='value for `event_type`'):
        p.add_slider_widget(callback=noop_callback, rng=[0, 10], event_type='foo')
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_slider_widget(callback=noop_callback, rng=[0, 10], style="modern", pass_widget=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_threshold(uniform, invert=True)
    p.add_mesh(uniform.outline())
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_threshold(uniform, invert=False)
    p.add_mesh(uniform.outline())
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_isovalue(uniform)
    p.close()

    p = pyvista.Plotter()
    title_height = np.random.random()
    s = p.add_slider_widget(
        callback=noop_callback, rng=[0, 10], style="classic", title_height=title_height
    )
    assert s.GetRepresentation().GetTitleHeight() == title_height
    p.close()

    p = pyvista.Plotter()
    title_opacity = np.random.random()
    s = p.add_slider_widget(
        callback=noop_callback, rng=[0, 10], style="classic", title_opacity=title_opacity
    )
    assert s.GetRepresentation().GetTitleProperty().GetOpacity() == title_opacity
    p.close()

    p = pyvista.Plotter()
    title_color = "red"
    s = p.add_slider_widget(
        callback=noop_callback, rng=[0, 10], style="classic", title_color=title_color
    )
    assert s.GetRepresentation().GetTitleProperty().GetColor() == pyvista.Color(title_color)
    p.close()

    p = pyvista.Plotter()
    fmt = "%0.9f"
    s = p.add_slider_widget(callback=noop_callback, rng=[0, 10], style="classic", fmt=fmt)
    assert s.GetRepresentation().GetLabelFormat() == fmt
    p.close()

    # custom width
    p = pyvista.Plotter()
    slider = p.add_slider_widget(
        callback=noop_callback, rng=[0, 10], fmt=fmt, tube_width=0.1, slider_width=0.2
    )
    assert slider.GetRepresentation().GetSliderWidth() == 0.2
    assert slider.GetRepresentation().GetTubeWidth() == 0.1
    p.close()


def test_widget_spline(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_spline_widget(callback=noop_callback)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    pts = np.array([[1, 5, 4], [2, 4, 9], [3, 6, 2]])
    with pytest.raises(ValueError, match='`initial_points` must be length `n_handles`'):
        p.add_spline_widget(callback=noop_callback, n_handles=4, initial_points=pts)
    p.add_spline_widget(callback=noop_callback, n_handles=3, initial_points=pts)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_spline_widget(callback=noop_callback, pass_widget=True, color=None, show_ribbon=True)
    p.close()

    p = pyvista.Plotter()
    p.add_mesh_slice_spline(uniform)
    p.close()


def test_widget_uniform(uniform):
    p = pyvista.Plotter()
    p.add_sphere_widget(callback=noop_callback, center=(0, 0, 0))
    p.close()

    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    p = pyvista.Plotter()
    p.add_sphere_widget(callback=noop_callback, center=nodes)
    p.close()


def test_widget_checkbox_button(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform)
    p.add_checkbox_button_widget(callback=noop_callback)
    p.close()


def test_widget_closed(uniform):
    pl = pyvista.Plotter()
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_checkbox_button_widget(callback=noop_callback)


@pytest.mark.needs_vtk_version(9, 1)
def test_add_camera_orientation_widget(uniform):
    p = pyvista.Plotter()
    p.add_camera_orientation_widget()
    assert p.camera_widgets
    p.close()
    assert not p.camera_widgets
