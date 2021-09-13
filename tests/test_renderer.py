import pytest

import pyvista
from pyvista.plotting import system_supports_plotting


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_camera_position():
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Sphere())
    plotter.show()
    assert isinstance(plotter.camera_position, pyvista.CameraPosition)


def test_plotter_camera_position():
    plotter = pyvista.Plotter()
    plotter.renderer.set_position([1, 1, 1])


def test_renderer_set_viewup():
    plotter = pyvista.Plotter()
    plotter.renderer.set_viewup([1, 1, 1])


def test_reset_camera():
    plotter = pyvista.Plotter()
    plotter.reset_camera(bounds=(-1, 1, -1, 1, -1, 1))


def test_layer():
    plotter = pyvista.Plotter()
    plotter.renderer.layer = 1
    assert plotter.renderer.layer == 1
    plotter.renderer.layer = 0
    assert plotter.renderer.layer == 0


@pytest.mark.parametrize('has_border', (True, False))
def test_border(has_border):
    border_color = (1, 1, 1)
    border_width = 1
    plotter = pyvista.Plotter(border=has_border, border_color=border_color,
                              border_width=border_width)
    assert plotter.renderer.has_border is has_border

    if has_border:
        assert plotter.renderer.border_color == border_color
    else:
        assert plotter.renderer.border_color is None

    if has_border:
        assert plotter.renderer.border_width == border_width
    else:
        assert plotter.renderer.border_width == 0
