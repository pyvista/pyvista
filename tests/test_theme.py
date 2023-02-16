import pytest
import vtk

import pyvista
from pyvista import colors
from pyvista.themes import DefaultTheme
from pyvista.utilities.misc import PyVistaDeprecationWarning


@pytest.fixture
def default_theme():
    return pyvista.themes.DefaultTheme()


def test_backwards_compatibility():
    try:
        color = (0.1, 0.4, 0.7)
        pyvista.rcParams['color'] = color
        assert pyvista.rcParams['color'] == pyvista.Color(color)

        # test nested values
        init_value = pyvista.rcParams['axes']['show']
        pyvista.rcParams['axes']['show'] = not init_value
        assert pyvista.rcParams['axes']['show'] is not init_value
    finally:
        # always return to testing theme
        pyvista.set_plot_theme('testing')


@pytest.mark.parametrize(
    'parm', [('enabled', True), ('occlusion_ratio', 0.5), ('number_of_peels', 2)]
)
def test_depth_peeling_config(default_theme, parm):
    attr, value = parm
    assert hasattr(default_theme.depth_peeling, attr)
    setattr(default_theme.depth_peeling, attr, value)
    assert getattr(default_theme.depth_peeling, attr) == value


def test_depth_peeling_eq(default_theme):
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.depth_peeling.enabled = not my_theme.depth_peeling.enabled
    assert my_theme.depth_peeling != default_theme.depth_peeling
    assert my_theme.depth_peeling != 1


@pytest.mark.parametrize(
    'parm',
    [
        ('color', (0.1, 0.1, 0.1)),
        ('line_width', 1),
        ('opacity', 1.0),
        ('feature_angle', 20),
        ('decimate', 0.5),
    ],
)
def test_silhouette_config(default_theme, parm):
    attr, value = parm
    assert hasattr(default_theme.silhouette, attr)
    setattr(default_theme.silhouette, attr, value)
    assert getattr(default_theme.silhouette, attr) == value


def test_depth_silhouette_eq(default_theme):
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.silhouette.opacity = 0.11111
    assert my_theme.silhouette != default_theme.silhouette
    assert my_theme.silhouette != 1


def test_depth_silhouette_opacity_outside_clamp(default_theme):
    my_theme = pyvista.themes.DefaultTheme()
    with pytest.raises(ValueError):
        my_theme.silhouette.opacity = 10
    with pytest.raises(ValueError):
        my_theme.silhouette.opacity = -1


@pytest.mark.parametrize(
    'parm',
    [
        ('slider_length', 0.03),
        ('slider_width', 0.02),
        ('slider_color', (0.5, 0.5, 0.3)),
        ('tube_width', 0.02),
        ('tube_color', (0.5, 0.5, 0.5)),
        ('cap_opacity', 0.5),
        ('cap_length', 0.02),
        ('cap_width', 0.04),
    ],
)
@pytest.mark.parametrize('style', ('modern', 'classic'))
def test_slider_style_config(default_theme, parm, style):
    attr, value = parm

    slider_style = getattr(default_theme.slider_styles, style)
    assert hasattr(slider_style, attr)
    setattr(slider_style, attr, value)
    assert getattr(slider_style, attr) == value


def test_slider_style_config_eq(default_theme):
    assert default_theme.slider_styles.modern != default_theme.slider_styles.classic
    assert default_theme.slider_styles.modern != 1


def test_slider_style_eq(default_theme):
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.slider_styles.modern.slider_length *= 2
    assert default_theme.slider_styles != my_theme.slider_styles


def test_invalid_color_str_single_char():
    with pytest.raises(ValueError):
        colors.Color('x')


def test_color_str():
    clr = colors.Color("k")
    assert (0.0, 0.0, 0.0) == clr
    clr = colors.Color("black")
    assert (0.0, 0.0, 0.0) == clr
    clr = colors.Color("white")
    assert (1.0, 1.0, 1.0) == clr
    with pytest.raises(ValueError):
        colors.Color('not a color')


def test_font():
    font = pyvista.parse_font_family('times')
    assert font == vtk.VTK_TIMES
    with pytest.raises(ValueError):
        pyvista.parse_font_family('not a font')


def test_font_eq(default_theme):
    defa_theme = pyvista.themes.DefaultTheme()
    assert defa_theme.font == default_theme.font

    paraview_theme = pyvista.themes.ParaViewTheme()
    assert paraview_theme.font != default_theme.font
    assert paraview_theme.font != 1


def test_font_family(default_theme):
    font = 'courier'
    default_theme.font.family = font
    assert default_theme.font.family == font

    with pytest.raises(ValueError):
        default_theme.font.family = 'bla'


def test_font_title_size(default_theme):
    default_theme.font.title_size = None
    assert default_theme.font.title_size is None


def test_font_label_size(default_theme):
    default_theme.font.label_size = None
    assert default_theme.font.label_size is None


def test_font_fmt(default_theme):
    fmt = '%.6e'
    default_theme.font.fmt = fmt
    assert default_theme.font.fmt == fmt


def test_axes_eq(default_theme):
    assert default_theme.axes == pyvista.themes.DefaultTheme().axes

    theme = pyvista.themes.DefaultTheme()
    theme.axes.box = True
    assert default_theme.axes != theme.axes
    assert default_theme.axes != 1


def test_theme_wrong_type(default_theme):
    with pytest.raises(TypeError):
        default_theme.font = None
    with pytest.raises(TypeError):
        default_theme.colorbar_horizontal = None
    with pytest.raises(TypeError):
        default_theme.colorbar_vertical = None
    with pytest.raises(TypeError):
        default_theme.depth_peeling = None
    with pytest.raises(TypeError):
        default_theme.silhouette = None
    with pytest.raises(TypeError):
        default_theme.slider_styles = None
    with pytest.raises(TypeError):
        default_theme.slider_styles.classic = None
    with pytest.raises(TypeError):
        default_theme.slider_styles.modern = None
    with pytest.raises(TypeError):
        default_theme.axes = None


def test_axes_box(default_theme):
    new_value = not default_theme.axes.box
    default_theme.axes.box = new_value
    assert default_theme.axes.box == new_value


def test_axes_show(default_theme):
    new_value = not default_theme.axes.show
    default_theme.axes.show = new_value
    assert default_theme.axes.show == new_value


def test_colorbar_eq(default_theme):
    theme = pyvista.themes.DefaultTheme()
    assert default_theme.colorbar_horizontal == theme.colorbar_horizontal

    assert default_theme.colorbar_horizontal != 1
    assert default_theme.colorbar_horizontal != theme.colorbar_vertical


def test_colorbar_height(default_theme):
    height = 0.3
    default_theme.colorbar_horizontal.height = height
    assert default_theme.colorbar_horizontal.height == height


def test_colorbar_width(default_theme):
    width = 0.3
    default_theme.colorbar_horizontal.width = width
    assert default_theme.colorbar_horizontal.width == width


def test_colorbar_position_x(default_theme):
    position_x = 0.3
    default_theme.colorbar_horizontal.position_x = position_x
    assert default_theme.colorbar_horizontal.position_x == position_x


def test_colorbar_position_y(default_theme):
    position_y = 0.3
    default_theme.colorbar_horizontal.position_y = position_y
    assert default_theme.colorbar_horizontal.position_y == position_y


@pytest.mark.parametrize('theme', pyvista.themes._NATIVE_THEMES)
def test_themes(theme):
    try:
        pyvista.set_plot_theme(theme.name)
        assert pyvista.global_theme == theme.value()
    finally:
        # always return to testing theme
        pyvista.set_plot_theme('testing')


def test_invalid_theme():
    with pytest.raises(ValueError):
        pyvista.set_plot_theme('this is not a valid theme')


def test_invalid_theme_type_error():
    with pytest.raises(TypeError):
        pyvista.set_plot_theme(1)


def test_set_theme():
    theme = pyvista.themes.DarkTheme()
    try:
        pyvista.set_plot_theme(theme)
        assert pyvista.global_theme == theme
    finally:
        # always return to testing theme
        pyvista.set_plot_theme('testing')


def test_invalid_load_theme(default_theme):
    with pytest.raises(TypeError):
        default_theme.load_theme(123)


def test_window_size(default_theme):
    with pytest.raises(ValueError):
        default_theme.window_size = [1, 2, 3]

    with pytest.raises(ValueError, match='Window size must be a positive value'):
        default_theme.window_size = [-1, -2]

    window_size = [1, 1]
    default_theme.window_size = window_size
    assert default_theme.window_size == window_size


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


def test_cmap(default_theme):
    cmap = 'jet'
    default_theme.cmap = cmap
    assert default_theme.cmap == cmap

    with pytest.raises(KeyError, match='not a color map'):
        default_theme.cmap = 'not a color map'


def test_volume_mapper(default_theme):
    assert hasattr(default_theme, 'volume_mapper')
    volume_mapper = 'gpu'
    default_theme.volume_mapper = volume_mapper
    assert default_theme.volume_mapper == volume_mapper

    with pytest.raises(ValueError, match='unknown'):
        default_theme.volume_mapper = 'invalid'


def test_set_hidden_line_removal(default_theme):
    default_theme.hidden_line_removal = True
    assert default_theme.hidden_line_removal is True
    default_theme.hidden_line_removal = False
    assert default_theme.hidden_line_removal is False


@pytest.mark.parametrize(
    'parm',
    [
        ('background', (0.1, 0.2, 0.3)),
        ('auto_close', False),
        ('notebook', False),
        ('full_screen', True),
        ('nan_color', (0.5, 0.5, 0.5)),
        ('edge_color', (1.0, 0.0, 0.0)),
        ('outline_color', (1.0, 0.0, 0.0)),
        ('floor_color', (1.0, 0.0, 0.0)),
        ('show_scalar_bar', False),
        ('lighting', False),
        ('interactive', False),
        ('render_points_as_spheres', True),
        ('transparent_background', True),
        ('title', 'test_title'),
        ('multi_samples', 10),
        ('multi_rendering_splitting_position', 0.1),
        ('smooth_shading', True),
        ('name', 'test_theme'),
        ('split_sharp_edges', True),
        ('sharp_edges_feature_angle', 45.0),
    ],
)
def test_theme_parm(default_theme, parm):
    attr, value = parm
    assert hasattr(default_theme, attr)
    setattr(default_theme, attr, value)
    assert getattr(default_theme, attr) == value


def test_theme_colorbar_orientation(default_theme):
    orient = 'vertical'
    default_theme.colorbar_orientation = orient
    assert default_theme.colorbar_orientation == orient

    with pytest.raises(ValueError):
        default_theme.colorbar_orientation = 'invalid'


def test_restore_defaults(default_theme):
    orig_value = default_theme.show_edges
    default_theme.show_edges = not orig_value
    default_theme.restore_defaults()
    assert default_theme.show_edges == orig_value


def test_repr(default_theme):
    rep = str(default_theme)
    assert 'Background' in rep
    assert default_theme.cmap in rep
    assert str(default_theme.colorbar_orientation) in rep
    assert default_theme._name.capitalize() in rep

    # verify that the key for each line in the repr is less than the minimum
    # key size. This makes sure that any new keys are either less than the size
    # of the key in the repr or the key length is increased
    for line in rep.splitlines():
        if ':' in line:
            pref, *rest = line.split(':', 1)
            assert pref.endswith(' '), f"Key str too long or need to raise key length:\n{pref!r}"


def test_theme_slots(default_theme):
    # verify we can't create an arbitrary attribute
    with pytest.raises(AttributeError, match='has no attribute'):
        default_theme.new_attr = 1


def test_theme_eq():
    defa_theme0 = pyvista.themes.DefaultTheme()
    defa_theme1 = pyvista.themes.DefaultTheme()
    assert defa_theme0 == defa_theme1
    dark_theme = pyvista.themes.DarkTheme()
    assert defa_theme0 != dark_theme

    # for coverage
    assert defa_theme0 != 'apple'


def test_plotter_set_theme():
    # test that the plotter theme is set to the new theme
    my_theme = pyvista.themes.DefaultTheme()
    my_theme.color = [1.0, 0.0, 0.0]
    pl = pyvista.Plotter(theme=my_theme)
    assert pl.theme.color == my_theme.color
    assert pyvista.global_theme.color != pl.theme.color

    pl = pyvista.Plotter()
    assert pl.theme == pyvista.global_theme
    pl.theme = my_theme
    assert pl.theme != pyvista.global_theme
    assert pl.theme == my_theme


def test_load_theme(tmpdir, default_theme):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.json'))
    pyvista.themes.DarkTheme().save(filename)
    loaded_theme = pyvista.load_theme(filename)
    assert loaded_theme == pyvista.themes.DarkTheme()

    default_theme.load_theme(filename)
    assert default_theme == pyvista.themes.DarkTheme()


def test_save_before_close_callback(tmpdir, default_theme):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.json'))
    dark_theme = pyvista.themes.DarkTheme()

    def fun(plotter):
        pass

    dark_theme.before_close_callback = fun
    assert dark_theme != pyvista.themes.DarkTheme()
    dark_theme.save(filename)

    # fun is stripped from the theme
    loaded_theme = pyvista.load_theme(filename)
    assert loaded_theme == pyvista.themes.DarkTheme()

    default_theme.load_theme(filename)
    assert default_theme == pyvista.themes.DarkTheme()


def test_anti_aliasing(default_theme):
    # test backwards compatibility
    with pytest.warns(PyVistaDeprecationWarning, match='is now a string'):
        default_theme.anti_aliasing = True
        pl = pyvista.Plotter(theme=default_theme)
        assert pl.renderer.GetUseFXAA()

    with pytest.raises(ValueError, match='anti_aliasing must be either'):
        default_theme.anti_aliasing = 'invalid value'

    with pytest.raises(TypeError, match='must be either'):
        default_theme.anti_aliasing = 42


def test_anti_aliasing_fxaa(default_theme):
    default_theme.anti_aliasing = 'fxaa'
    assert default_theme.anti_aliasing == 'fxaa'
    pl = pyvista.Plotter(theme=default_theme)
    assert pl.renderer.GetUseFXAA()


def test_anti_aliasing_ssaa(default_theme):
    # default should is not enabled
    if default_theme.anti_aliasing != 'ssaa':
        pl = pyvista.Plotter(theme=default_theme)
        assert 'vtkSSAAPass' not in pl.renderer._render_passes._passes

    default_theme.anti_aliasing = 'ssaa'
    assert default_theme.anti_aliasing == 'ssaa'
    pl = pyvista.Plotter(theme=default_theme)
    assert 'vtkSSAAPass' in pl.renderer._render_passes._passes


def test_anti_aliasing_msaa(default_theme):
    if default_theme.anti_aliasing != 'msaa':
        pl = pyvista.Plotter(theme=default_theme)
        assert pl.render_window.GetMultiSamples() == 0

    default_theme.anti_aliasing = 'msaa'
    default_theme.multi_samples = 4
    assert default_theme.anti_aliasing == 'msaa'
    pl = pyvista.Plotter(theme=default_theme)
    assert pl.render_window.GetMultiSamples() == default_theme.multi_samples


def test_antialiasing_deprecation(default_theme):
    with pytest.warns(PyVistaDeprecationWarning, match='anti_aliasing'):
        default_theme.antialiasing
    with pytest.warns(PyVistaDeprecationWarning, match='anti_aliasing'):
        default_theme.antialiasing = True


def test_above_range_color(default_theme):
    default_theme.above_range_color = 'r'
    assert isinstance(default_theme.above_range_color, pyvista.Color)


def test_below_range_color(default_theme):
    default_theme.below_range_color = 'b'
    assert isinstance(default_theme.below_range_color, pyvista.Color)


def test_user_theme():
    class MyTheme(DefaultTheme):
        def __init__(self):
            """Initialize the theme."""
            super().__init__()
            self.background = 'lightgrey'
            self.color = '#1f77b4'

            self.lighting_params.interpolation = 'Phong'
            self.lighting_params.ambient = 0.15
            self.lighting_params.diffuse = 0.45
            self.lighting_params.specular = 0.85
            self.lighting_params.roughness = 0.25  # PBR
            self.lighting_params.metallic = 0.35  # PBR

            self.smooth_shading = True
            self.render_lines_as_tubes = True
            self.line_width = 8
            self.point_size = 9

    theme = MyTheme()
    sphere = pyvista.Sphere()
    lines = sphere.extract_all_edges()
    points = pyvista.PolyData(sphere.points)
    try:
        pyvista.set_plot_theme(theme)

        pl = pyvista.Plotter()
        assert pl.background_color == theme.background
        sactor = pl.add_mesh(sphere)
        assert sactor.prop.color == theme.color
        assert sactor.prop.interpolation.value == theme.lighting_params.interpolation
        assert sactor.prop.ambient == theme.lighting_params.ambient
        assert sactor.prop.diffuse == theme.lighting_params.diffuse
        assert sactor.prop.specular == theme.lighting_params.specular

        lactor = pl.add_mesh(lines)
        assert lactor.prop.render_lines_as_tubes == theme.render_lines_as_tubes
        assert lactor.prop.line_width == theme.line_width

        pactor = pl.add_mesh(points)
        assert pactor.prop.point_size == theme.point_size

        if pyvista._vtk.VTK9:
            pl = pyvista.Plotter()
            sactor = pl.add_mesh(sphere, pbr=True)
            assert sactor.prop.roughness == theme.lighting_params.roughness
            assert sactor.prop.metallic == theme.lighting_params.metallic

    finally:
        # always return to testing theme
        pyvista.set_plot_theme('testing')
