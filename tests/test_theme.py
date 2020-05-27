import pytest
import vtk

import pyvista
from pyvista import colors


def test_invalid_color_str_single_char():
    with pytest.raises(ValueError):
        colors.string_to_rgb('x')


def test_color_str():
    clr = colors.string_to_rgb("k")
    assert (0.0, 0.0, 0.0) == clr
    clr = colors.string_to_rgb("black")
    assert (0.0, 0.0, 0.0) == clr
    clr = colors.string_to_rgb("white")
    assert (1.0, 1.0, 1.0) == clr
    with pytest.raises(ValueError):
        colors.string_to_rgb('not a color')


def test_font():
    font = pyvista.parse_font_family('times')
    assert font == vtk.VTK_TIMES
    with pytest.raises(ValueError):
        pyvista.parse_font_family('not a font')
