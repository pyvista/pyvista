"""This test module tests any functionality related to the Charts API."""
import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.plotting import charts


def test_pen():
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    w_thin, w_thick = 2, 10
    s_dash, s_dot, s_inv = "--", ":", "|"
    assert s_inv not in charts.Pen.LINE_STYLES, "New line styles added? Change this test."

    # Test constructor arguments
    pen = charts.Pen(color=c_red, width=w_thin, style=s_dash)
    assert np.allclose(pen.color, c_red) and np.isclose(pen.width, w_thin) and pen.style == s_dash

    # Test properties
    pen.color = c_blue
    color = [0.0, 0.0, 0.0]
    pen.GetColorF(color)
    color.append(pen.GetOpacity() / 255)
    assert np.allclose(pen.color, c_blue) and np.allclose(color, c_blue)

    pen.width = w_thick
    assert np.isclose(pen.width, w_thick) and np.isclose(pen.GetWidth(), w_thick)

    pen.style = s_dot
    assert pen.style == s_dot and pen.GetLineType() == charts.Pen.LINE_STYLES[s_dot]
    with pytest.raises(ValueError):
        pen.style = s_inv


def test_wrapping():
    width = 5
    # Test wrapping of VTK Pen object
    vtkPen = pyvista._vtk.vtkPen()
    wrappedPen = charts.Pen(_wrap=vtkPen)
    assert wrappedPen.__this__ == vtkPen.__this__
    assert wrappedPen.width == vtkPen.GetWidth()
    wrappedPen.width = width
    assert wrappedPen.width == vtkPen.GetWidth() and vtkPen.GetWidth() == width


def test_brush():
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    t_masonry, t_puppy = examples.download_masonry_texture(), examples.download_puppy_texture()

    # Test constructor arguments
    brush = charts.Brush(color=c_red, texture=t_masonry)
    assert np.allclose(brush.color, c_red) and np.allclose(brush.texture.to_array(), t_masonry.to_array())

    # Test properties
    brush.color = c_blue
    color = [0.0, 0.0, 0.0, 0.0]
    brush.GetColorF(color)
    assert np.allclose(brush.color, c_blue) and np.allclose(color, c_blue)

    brush.texture = t_puppy
    t = pyvista.Texture(brush.GetTexture())
    assert np.allclose(brush.texture.to_array(), t_puppy.to_array()) and np.allclose(t.to_array(), t_puppy.to_array())

    brush.texture_interpolate = False
    NEAREST = 0x01
    assert not brush.texture_interpolate and brush.GetTextureProperties() & NEAREST

    brush.texture_repeat = True
    REPEAT = 0x08
    assert brush.texture_repeat and brush.GetTextureProperties() & REPEAT
