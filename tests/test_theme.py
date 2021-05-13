import pytest
import vtk

import pyvista
from pyvista import colors


@pytest.fixture
def default_theme():
    return pyvista.themes.DefaultTheme()


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


@pytest.mark.parametrize('theme', pyvista.themes.ALLOWED_THEMES)
def test_themes(theme):
    try:
        pyvista.set_plot_theme(theme.name)
        assert pyvista.global_theme == theme.value()
    finally:
        # always return to testing theme
        pyvista.set_plot_theme('testing')


def test_invalid_theme():
    with pytest.raises(KeyError):
        pyvista.set_plot_theme('this is not a valid theme')


def test_background(default_theme):
    color = [0.1, 0.2, 0.3]
    default_theme.background = color
    assert default_theme.background == color


def test_auto_close(default_theme):
    auto_close = not default_theme.auto_close
    default_theme.auto_close = auto_close
    assert default_theme.auto_close == auto_close


def test_font(default_theme):
    font = 'courier'
    default_theme.auto_close = font
    assert default_theme.auto_close == font

    # with pytest.raises(ValueError):
        # default_theme.auto_close = 'bla'


def test_window_size(default_theme):
    with pytest.raises(ValueError):
        default_theme.window_size = [1, 2, 3]

    with pytest.raises(ValueError, match='Window size must be a positive value'):
        default_theme.window_size = [-1, -2]

    window_size = [1, 1]
    default_theme.notebook = window_size
    assert default_theme.notebook == window_size


def test_notebook(default_theme):
    notebook = not default_theme.notebook
    default_theme.notebook = notebook
    assert default_theme.notebook == notebook


def test_full_screen(default_theme):
    full_screen = not default_theme.full_screen
    default_theme.full_screen = full_screen
    assert default_theme.full_screen == full_screen


def test_camera(default_theme):
    with pytest.raises(TypeError):
        default_theme.camera = [1, 0, 0]
    
    with pytest.raises(KeyError, match='Expected the "viewup"'):
        default_theme.camera = {'position': [1, 0, 0]}

    with pytest.raises(KeyError, match='Expected the "position"'):
        default_theme.camera = {'viewup': [1, 0, 0]}

    camera = {'position': [1, 0, 0], 'viewup': [1, 0, 0]}
    default_theme.camera = camera
    assert default_theme.camera == camera


def test_repr():
    theme = pyvista.themes.DefaultTheme()
    rep = str(theme)
    assert 'Background' in rep
    assert theme.cmap in rep
    assert str(theme.colorbar_orientation) in rep
    assert theme._name.capitalize() in rep


def test_theme_eq():
    defa_theme0 = pyvista.themes.DefaultTheme()
    defa_theme1 = pyvista.themes.DefaultTheme()
    assert defa_theme0 == defa_theme1
    dark_theme = pyvista.themes.DarkTheme()
    assert defa_theme0 != dark_theme

    # for coverage
    assert defa_theme0 != 'apple', 'wrong type test failed'


def test_plotter_set_theme():
    # test that the plotter theme is set to the new theme
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.color = [1, 0, 0]
    pl = pyvista.Plotter(theme=my_theme)
    assert pl.theme.color == my_theme.color

    # test that the plotter theme isn't overridden by the global theme
    try:
        other_color = [0, 0, 0]
        pyvista.global_theme.color = other_color
        assert pyvista.global_theme.color == other_color
        assert pl.theme.color == my_theme.color
    finally:
        # need "finally" here if the test fails and we mess up the
        # global defaults
        pyvista.global_theme.load_theme(pyvista.themes._TestingTheme())
