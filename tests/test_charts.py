"""Test charting functionality"""

import platform

import pytest
import numpy as np

import pyvista
from pyvista.plotting import charts
from pyvista import examples

skip_mac = pytest.mark.skipif(platform.system() == 'Darwin',
                              reason='MacOS CI fails when downloading examples')


def test_pen():
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    w_thin, w_thick = 2, 10
    s_dash, s_dot, s_inv = "--", ":", "|"
    assert s_inv not in charts.Pen.LINE_STYLES, "New line styles added? Change this test."

    # Test constructor arguments
    pen = charts.Pen(color=c_red, width=w_thin, style=s_dash)
    assert np.allclose(pen.color, c_red)
    assert np.isclose(pen.width, w_thin)
    assert pen.style == s_dash

    # Test properties
    pen.color = c_blue
    color = [0.0, 0.0, 0.0]
    pen.GetColorF(color)
    color.append(pen.GetOpacity() / 255)
    assert np.allclose(pen.color, c_blue)
    assert np.allclose(color, c_blue)

    pen.width = w_thick
    assert np.isclose(pen.width, w_thick)
    assert np.isclose(pen.GetWidth(), w_thick)

    pen.style = s_dot
    assert pen.style == s_dot
    assert pen.GetLineType() == charts.Pen.LINE_STYLES[s_dot]["id"]
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
    assert wrappedPen.width == vtkPen.GetWidth()
    assert vtkPen.GetWidth() == width


@skip_mac
def test_brush():
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    t_masonry = examples.download_masonry_texture()
    t_puppy = examples.download_puppy_texture()

    # Test constructor arguments
    brush = charts.Brush(color=c_red, texture=t_masonry)
    assert np.allclose(brush.color, c_red)
    assert np.allclose(brush.texture.to_array(), t_masonry.to_array())

    # Test properties
    brush.color = c_blue
    color = [0.0, 0.0, 0.0, 0.0]
    brush.GetColorF(color)
    assert np.allclose(brush.color, c_blue)
    assert np.allclose(color, c_blue)

    brush.texture = t_puppy
    t = pyvista.Texture(brush.GetTexture())
    assert np.allclose(brush.texture.to_array(), t_puppy.to_array())
    assert np.allclose(t.to_array(), t_puppy.to_array())

    brush.texture_interpolate = False
    assert not brush.texture_interpolate

    NEAREST = 0x01
    assert brush.GetTextureProperties() & NEAREST

    brush.texture_repeat = True
    assert brush.texture_repeat
    REPEAT = 0x08
    assert brush.GetTextureProperties() & REPEAT
