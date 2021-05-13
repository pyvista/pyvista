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



# @pytest.mark.parametrize('theme', ['paraview', 'document', 'night', 'default'])
# def test_themes(theme):
#     pyvista.set_plot_theme(theme)
#     if theme != 'default':
#         assert pyvista.rcParams != pyvista.DEFAULT_THEME
#         pyvista.set_plot_theme('default')
#     assert pyvista.rcParams == pyvista.DEFAULT_THEME

#     # always return to testing theme
#     pyvista.set_plot_theme('testing')


# def test_invalid_theme():
#     with pytest.raises(ValueError, match='Invalid theme'):
#         pyvista.set_plot_theme('this is not a valid theme')
