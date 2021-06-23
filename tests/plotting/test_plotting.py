"""
See the image regression notes in doc/extras/developer_notes.rst
"""
import time
import platform
import warnings
import inspect
import pathlib
import os
from weakref import proxy
from pathlib import Path

from PIL import Image
import imageio
import numpy as np
import pytest
import vtk

import pyvista
from pyvista._vtk import VTK9
from pyvista import examples
from pyvista.plotting import system_supports_plotting
from pyvista.plotting.plotting import SUPPORTED_FORMATS
from pyvista.core.errors import DeprecationError


ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        imageio.plugins.ffmpeg.download()
except:
    ffmpeg_failed = True

try:
    from vtkmodules.vtkCommonCore import vtkVersion
    vtk_dev = len(str(vtkVersion().GetVTKBuildVersion())) > 2
except:
    vtk_dev = False

# These tests fail with mesa opengl on windows
skip_windows_dev_whl = pytest.mark.skipif(os.name == 'nt' and vtk_dev,
                                          reason='Test fails on Windows with VTK dev wheels')


# Reset image cache with new images
glb_reset_image_cache = False
IMAGE_CACHE_DIR = os.path.join(Path(__file__).parent.absolute(), 'image_cache')
if not os.path.isdir(IMAGE_CACHE_DIR):
    os.mkdir(IMAGE_CACHE_DIR)

# always set on azure CI
AZURE_CI_WINDOWS = os.environ.get('AZURE_CI_WINDOWS', 'false').lower() == 'true'

skip_no_plotting = pytest.mark.skipif(not system_supports_plotting(),
                                      reason="Test requires system to support plotting")

skip_not_vtk9 = pytest.mark.skipif(not VTK9, reason="Test requires >=VTK v9")

# Normal image warning/error thresholds (assumes using use_vtk)
IMAGE_REGRESSION_ERROR = 500  # major differences
IMAGE_REGRESSION_WARNING = 200  # minor differences

# Image regression warning/error thresholds for releases after 9.0.1
HIGH_VARIANCE_TESTS = {'test_pbr', 'test_set_viewup', 'test_add_title'}
VER_IMAGE_REGRESSION_ERROR = 1000
VER_IMAGE_REGRESSION_WARNING = 1000


# this must be a session fixture to ensure this runs before any other test
@pytest.fixture(scope="session", autouse=True)
def get_cmd_opt(pytestconfig):
    global glb_reset_image_cache, glb_ignore_image_cache
    glb_reset_image_cache = pytestconfig.getoption('reset_image_cache')
    glb_ignore_image_cache = pytestconfig.getoption('ignore_image_cache')


def verify_cache_image(plotter):
    """Either store or validate an image.

    This is function should only be called within a pytest
    environment.  Pass it to either the ``Plotter.show()`` or the
    ``pyvista.plot()`` functions as the before_close_callback keyword
    arg.

    Assign this only once for each test you'd like to validate the
    previous image of.  This will not work with parameterized tests.

    Example Usage:
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)

    """
    global glb_reset_image_cache, glb_ignore_image_cache

    # Image cache is only valid for VTK9 on Linux
    if not VTK9 or platform.system() != 'Linux':
        return

    # since each test must contain a unique name, we can simply
    # use the function test to name the image
    stack = inspect.stack()
    test_name = None
    for item in stack:
        if item.function == 'check_gc':
            return
        if item.function[:5] == 'test_':
            test_name = item.function
            break
    if item.function in HIGH_VARIANCE_TESTS:
        allowed_error = VER_IMAGE_REGRESSION_ERROR
        allowed_warning = VER_IMAGE_REGRESSION_WARNING
    else:
        allowed_error = IMAGE_REGRESSION_ERROR
        allowed_warning = IMAGE_REGRESSION_WARNING

    if test_name is None:
        raise RuntimeError('Unable to identify calling test function.  This function '
                           'should only be used within a pytest environment.')

    # cached image name
    image_filename = os.path.join(IMAGE_CACHE_DIR, test_name[5:] + '.png')

    # simply save the last screenshot if it doesn't exist or the cache
    # is being reset.
    if glb_reset_image_cache or not os.path.isfile(image_filename):
        return plotter.screenshot(image_filename)

    if glb_ignore_image_cache:
        return

    # otherwise, compare with the existing cached image
    error = pyvista.compare_images(image_filename, plotter)
    if error > allowed_error:
        raise RuntimeError('Exceeded image regression error of '
                           f'{IMAGE_REGRESSION_ERROR} with an image error of '
                           f'{error}')
    if error > allowed_warning:
        warnings.warn('Exceeded image regression warning of '
                      f'{IMAGE_REGRESSION_WARNING} with an image error of '
                      f'{error}')


@skip_not_vtk9
@skip_no_plotting
@skip_windows_dev_whl
@pytest.mark.skipif(AZURE_CI_WINDOWS, reason="Windows CI testing segfaults on pbr")
def test_pbr(sphere):
    """Test PBR rendering"""
    texture = examples.load_globe_texture()

    pl = pyvista.Plotter(lighting=None)
    pl.set_environment_texture(texture)
    pl.add_light(pyvista.Light())
    pl.add_mesh(sphere,
                color='w',
                pbr=True, metallic=0.8, roughness=0.2,
                smooth_shading=True,
                diffuse=1)
    pl.add_mesh(pyvista.Sphere(center=(0, 0, 1)),
                color='w',
                pbr=True, metallic=0.0, roughness=1.0,
                smooth_shading=True,
                diffuse=1)
    pl.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_pyvista_ndarray(sphere):
    # verify we can plot pyvista_ndarray
    pyvista.plot(sphere.points)

    plotter = pyvista.Plotter()
    plotter.add_points(sphere.points)
    plotter.add_points(sphere.points + 1)
    plotter.show()


@skip_no_plotting
def test_plot_increment_point_size():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    pl = pyvista.Plotter()
    pl.add_points(points + 1)
    pl.add_lines(points)
    pl.increment_point_size_and_line_width(5)
    pl.show(before_close_callback=verify_cache_image)


@skip_not_vtk9
@skip_no_plotting
def test_plot_update(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    pl.update()
    time.sleep(0.1)
    pl.update()
    pl.update(force_redraw=True)
    pl.close()


@skip_no_plotting
def test_plot(sphere, tmpdir):
    tmp_dir = tmpdir.mkdir("tmpdir2")
    filename = str(tmp_dir.join('tmp.png'))
    scalars = np.arange(sphere.n_points)
    cpos, img = pyvista.plot(sphere,
                             full_screen=True,
                             text='this is a sphere',
                             show_bounds=True,
                             color='r',
                             style='wireframe',
                             line_width=2,
                             scalars=scalars,
                             flip_scalars=True,
                             cmap='bwr',
                             interpolate_before_map=True,
                             screenshot=filename,
                             return_img=True,
                             before_close_callback=verify_cache_image)
    assert isinstance(cpos, pyvista.CameraPosition)
    assert isinstance(img, np.ndarray)
    assert os.path.isfile(filename)

    filename = pathlib.Path(str(tmp_dir.join('tmp2.png')))
    cpos = pyvista.plot(sphere, screenshot=filename)

    # Ensure it added a PNG extension by default
    assert filename.with_suffix(".png").is_file()

    # test invalid extension
    with pytest.raises(ValueError):
        filename = pathlib.Path(str(tmp_dir.join('tmp3.foo')))
        pyvista.plot(sphere, screenshot=filename)


@skip_no_plotting
def test_add_title():
    plotter = pyvista.Plotter()
    plotter.add_title('Plot Title')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_invalid_style(sphere):
    with pytest.raises(ValueError):
        pyvista.plot(sphere, style='not a style')


@skip_no_plotting
@pytest.mark.parametrize('interaction, kwargs', [
    ('trackball', {}),
    ('trackball_actor', {}),
    ('image', {}),
    ('joystick', {}),
    ('joystick_actor', {}),
    ('zoom', {}),
    ('terrain', {}),
    ('terrain', {'mouse_wheel_zooms': True, 'shift_pans': True}),
    ('rubber_band', {}),
    ('rubber_band_2d', {}),
])
def test_interactor_style(sphere, interaction, kwargs):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    getattr(plotter, f'enable_{interaction}_style')(**kwargs)
    assert plotter.iren._style_class is not None
    plotter.close()


def test_lighting_disable_3_lights():
    with pytest.raises(DeprecationError):
        pyvista.Plotter().disable_3_lights()


@skip_no_plotting
def test_lighting_enable_three_lights(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)

    plotter.enable_3_lights()
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.on

    assert lights[0].intensity == 1.0
    assert lights[1].intensity == 0.6
    assert lights[2].intensity == 0.5

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_add_manual_light(sphere):
    plotter = pyvista.Plotter(lighting=None)
    plotter.add_mesh(sphere)

    # test manual light addition
    light = pyvista.Light()
    plotter.add_light(light)
    assert plotter.renderer.lights == [light]

    # failing case
    with pytest.raises(TypeError):
        plotter.add_light('invalid')

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_remove_manual_light(sphere):
    plotter = pyvista.Plotter(lighting=None)
    plotter.add_mesh(sphere)
    plotter.add_light(pyvista.Light())

    # test light removal
    plotter.remove_all_lights()
    assert not plotter.renderer.lights

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_subplots(sphere):
    plotter = pyvista.Plotter(shape='1|1')
    plotter.add_mesh(sphere)
    renderers = plotter.renderers

    light = pyvista.Light()
    plotter.remove_all_lights()
    for renderer in renderers:
        assert not renderer.lights

    plotter.subplot(0)
    plotter.add_light(light, only_active=True)
    assert renderers[0].lights and not renderers[1].lights
    plotter.add_light(light, only_active=False)
    assert renderers[0].lights and renderers[1].lights
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Sphere())
    plotter.remove_all_lights(only_active=True)
    assert renderers[0].lights and not renderers[1].lights

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_init_light_kit(sphere):
    plotter = pyvista.Plotter(lighting='light kit')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert len(lights) == 5
    assert lights[0].light_type == pyvista.Light.HEADLIGHT
    for light in lights[1:]:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_init_three_lights(sphere):
    plotter = pyvista.Plotter(lighting='three lights')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert len(lights) == 3
    for light in lights:
        assert light.light_type == light.CAMERA_LIGHT
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_lighting_init_none(sphere):
    # ``None`` already tested above
    plotter = pyvista.Plotter(lighting='none')
    plotter.add_mesh(sphere)
    lights = plotter.renderer.lights
    assert not lights
    plotter.show(before_close_callback=verify_cache_image)


def test_lighting_init_invalid():
    with pytest.raises(ValueError):
        pyvista.Plotter(lighting='invalid')


@skip_no_plotting
def test_plotter_shape_invalid():
    # wrong size
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(1,))
    # not positive
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(1, 0))
    with pytest.raises(ValueError):
        pyvista.Plotter(shape=(0, 2))
    # not a sequence
    with pytest.raises(TypeError):
        pyvista.Plotter(shape={1, 2})


@skip_no_plotting
def test_plot_bounds_axes_with_no_data():
    plotter = pyvista.Plotter()
    plotter.show_bounds()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_show_grid(sphere):
    plotter = pyvista.Plotter()
    plotter.show_grid()
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)


cpos_param = [[(2.0, 5.0, 13.0),
              (0.0, 0.0, 0.0),
              (-0.7, -0.5, 0.3)],
             [-1, 2, -5],  # trigger view vector
             [1.0, 2.0, 3.0],
]
cpos_param.extend(pyvista.plotting.Renderer.CAMERA_STR_ATTR_MAP)
@skip_no_plotting
@pytest.mark.parametrize('cpos', cpos_param)
def test_set_camera_position(cpos, sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.camera_position = cpos
    plotter.show()


@skip_no_plotting
@pytest.mark.parametrize('cpos', [[(2.0, 5.0),
                                   (0.0, 0.0, 0.0),
                                   (-0.7, -0.5, 0.3)],
                                  [-1, 2],
                                  [(1,2,3)],
                                  'notvalid'])
def test_set_camera_position_invalid(cpos, sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(pyvista.core.errors.InvalidCameraError):
        plotter.camera_position = cpos


@skip_no_plotting
def test_parallel_projection():
    plotter = pyvista.Plotter()
    assert isinstance(plotter.parallel_projection, bool)


@skip_no_plotting
@pytest.mark.parametrize("state", [True, False])
def test_set_parallel_projection(state):
    plotter = pyvista.Plotter()
    plotter.parallel_projection = state
    assert plotter.parallel_projection == state


@skip_no_plotting
def test_parallel_scale():
    plotter = pyvista.Plotter()
    assert isinstance(plotter.parallel_scale, float)


@skip_no_plotting
@pytest.mark.parametrize("value", [1, 1.5, 0.3, 10])
def test_set_parallel_scale(value):
    plotter = pyvista.Plotter()
    plotter.parallel_scale = value
    assert plotter.parallel_scale == value


@skip_no_plotting
def test_set_parallel_scale_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(TypeError):
        plotter.parallel_scale = "invalid"


@skip_no_plotting
def test_plot_no_active_scalars(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(5))
    with pytest.raises(ValueError):
        plotter.update_scalars(np.arange(sphere.n_faces))


@skip_no_plotting
def test_plot_show_bounds(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(show_xaxis=False,
                        show_yaxis=False,
                        show_zaxis=False,
                        show_xlabels=False,
                        show_ylabels=False,
                        show_zlabels=False,
                        use_2d=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_label_fmt(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.show_bounds(xlabel='My X', fmt=r'%.3f')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.parametrize('grid', [True, 'both', 'front', 'back'])
@pytest.mark.parametrize('location', ['all', 'origin', 'outer', 'front', 'back'])
def test_plot_show_bounds_params(grid, location):
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Cube())
    plotter.show_bounds(grid=grid, ticks='inside', location=location)
    plotter.show_bounds(grid=grid, ticks='outside', location=location)
    plotter.show_bounds(grid=grid, ticks='both', location=location)
    plotter.show()


@skip_no_plotting
def test_plot_silhouette_fail(hexbeam):
    plotter = pyvista.Plotter()
    with pytest.raises(TypeError, match="Expected type is `PolyData`"):
        plotter.add_mesh(hexbeam, silhouette=True)


@skip_no_plotting
def test_plot_no_silhouette(tri_cylinder):
    # silhouette=False
    plotter = pyvista.Plotter()
    plotter.add_mesh(tri_cylinder)
    assert len(list(plotter.renderer.GetActors())) == 1  # only cylinder
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_silhouette(tri_cylinder):
    # silhouette=True and default properties
    plotter = pyvista.Plotter()
    plotter.add_mesh(tri_cylinder, silhouette=True)
    actors = list(plotter.renderer.GetActors())
    assert len(actors) == 2  # cylinder + silhouette
    actor = actors[0]  # get silhouette actor
    props = actor.GetProperty()
    assert props.GetColor() == pyvista.parse_color(pyvista.global_theme.silhouette.color)
    assert props.GetOpacity() == pyvista.global_theme.silhouette.opacity
    assert props.GetLineWidth() == pyvista.global_theme.silhouette.line_width
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_silhouette_options(tri_cylinder):
    # cover other properties
    plotter = pyvista.Plotter()
    plotter.add_mesh(tri_cylinder, silhouette=dict(decimate=None, feature_angle=20))
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plotter_scale(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.set_scale(10, 10, 10)
    assert plotter.scale == [10, 10, 10]
    plotter.set_scale(5.0)
    plotter.set_scale(yscale=6.0)
    plotter.set_scale(zscale=9.0)
    assert plotter.scale == [5.0, 6.0, 9.0]
    plotter.scale = [1.0, 4.0, 2.0]
    assert plotter.scale == [1.0, 4.0, 2.0]
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_add_scalar_bar(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_scalar_bar(label_font_size=10, title_font_size=20, title='woa',
                           interactive=True, vertical=True)
    plotter.add_scalar_bar(background_color='white', n_colors=256)
    assert isinstance(plotter.scalar_bar, vtk.vtkScalarBarActor)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_invalid_add_scalar_bar():
    with pytest.raises(AttributeError):
        plotter = pyvista.Plotter()
        plotter.add_scalar_bar()


@skip_no_plotting
def test_plot_list():
    sphere_a = pyvista.Sphere(0.5)
    sphere_b = pyvista.Sphere(1.0)
    sphere_c = pyvista.Sphere(2.0)
    pyvista.plot([sphere_a, sphere_b, sphere_c],
                 style='wireframe',
                 before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_lines_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(TypeError):
        plotter.add_lines(range(10))


@skip_no_plotting
def test_open_gif_invalid():
    plotter = pyvista.Plotter()
    with pytest.raises(ValueError):
        plotter.open_gif('file.abs')


@pytest.mark.skipif(ffmpeg_failed, reason="Requires imageio-ffmpeg")
@skip_no_plotting
def test_make_movie(sphere):
    # Make temporary file
    filename = os.path.join(pyvista.USER_DATA_PATH, 'tmp.mp4')

    movie_sphere = sphere.copy()
    plotter = pyvista.Plotter()
    plotter.open_movie(filename)
    actor = plotter.add_axes_at_origin()
    plotter.remove_actor(actor, reset_camera=False, render=True)
    plotter.add_mesh(movie_sphere,
                     scalars=np.random.random(movie_sphere.n_faces))
    plotter.show(auto_close=False, window_size=[304, 304])
    plotter.set_focus([0, 0, 0])
    for i in range(3):  # limiting number of frames to write for speed
        plotter.write_frame()
        random_points = np.random.random(movie_sphere.points.shape)
        movie_sphere.points[:] = random_points*0.01 + movie_sphere.points*0.99
        movie_sphere.points[:] -= movie_sphere.points.mean(0)
        scalars = np.random.random(movie_sphere.n_faces)
        plotter.update_scalars(scalars)

    # remove file
    plotter.close()
    os.remove(filename)  # verifies that the plotter has closed


@skip_no_plotting
def test_add_legend(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    with pytest.raises(ValueError):
        plotter.add_legend()
    legend_labels = [['sphere', 'r']]
    plotter.add_legend(labels=legend_labels, border=True, bcolor=None,
                       size=[0.1, 0.1])
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_legend_origin(sphere):
    """Ensure the origin parameter of `add_legend` affects origin position."""
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    origin = [0, 0]
    legend = plotter.add_legend(labels=legend_labels, border=True, bcolor=None,
                                size=[0.1, 0.1], origin=origin)
    assert list(origin) == list(legend.GetPosition())


@skip_no_plotting
def test_bad_legend_origin_and_size(sphere):
    """Ensure bad parameters to origin/size raise ValueErrors."""
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    legend_labels = [['sphere', 'r']]
    # test incorrect lengths
    with pytest.raises(ValueError, match='origin'):
        plotter.add_legend(labels=legend_labels, origin=(1, 2, 3))
    with pytest.raises(ValueError, match='size'):
        plotter.add_legend(labels=legend_labels, size=[])
    # test non-sequences also raise
    with pytest.raises(ValueError, match='origin'):
        plotter.add_legend(labels=legend_labels, origin=len)
    with pytest.raises(ValueError, match='size'):
        plotter.add_legend(labels=legend_labels, size=type)


@skip_no_plotting
def test_add_axes_twice():
    plotter = pyvista.Plotter()
    plotter.add_axes()
    plotter.add_axes(interactive=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_hide_axes():
    plotter = pyvista.Plotter()
    plotter.add_axes()
    plotter.hide_axes()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_show_axes_all():
    plotter = pyvista.Plotter()
    plotter.show_axes_all()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_hide_axes_all():
    plotter = pyvista.Plotter()
    plotter.hide_axes_all()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_isometric_view_interactive(sphere):
    plotter_iso = pyvista.Plotter()
    plotter_iso.add_mesh(sphere)
    plotter_iso.camera_position = 'xy'
    cpos_old = plotter_iso.camera_position
    plotter_iso.isometric_view_interactive()
    assert not plotter_iso.camera_position == cpos_old


@skip_no_plotting
def test_add_point_labels():
    plotter = pyvista.Plotter()

    # cannot use random points with image regression
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0.5, 0.5, 0.5],
                       [1, 1, 1]])
    n = points.shape[0]

    with pytest.raises(ValueError):
        plotter.add_point_labels(points, range(n - 1))

    plotter.add_point_labels(points, range(n), show_points=True,
                             point_color='r', point_size=10)
    plotter.add_point_labels(points - 1, range(n), show_points=False,
                             point_color='r', point_size=10)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.parametrize('always_visible', [False, True])
def test_add_point_labels_always_visible(always_visible):
    # just make sure it runs without exception
    plotter = pyvista.Plotter()
    plotter.add_point_labels(
        np.array([[0, 0, 0]]), ['hello world'], always_visible=always_visible)
    plotter.show()


@skip_no_plotting
def test_set_background():
    plotter = pyvista.Plotter()
    plotter.set_background('k')
    plotter.background_color = "yellow"
    plotter.set_background([0, 0, 0], top=[1, 1, 1])  # Gradient
    plotter.background_color
    plotter.show()

    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.set_background('orange')
    for renderer in plotter.renderers:
        assert renderer.GetBackground() == pyvista.parse_color('orange')
    plotter.show()

    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.subplot(0, 1)
    plotter.set_background('orange', all_renderers=False)
    assert plotter.renderers[0].GetBackground() != pyvista.parse_color('orange')
    assert plotter.renderers[1].GetBackground() == pyvista.parse_color('orange')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_points():
    plotter = pyvista.Plotter()

    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 0],
                       [0.5, 0.5, 0.5],
                       [1, 1, 1]])
    n = points.shape[0]

    plotter.add_points(points, scalars=np.arange(n), cmap=None, flip_scalars=True,
                       show_scalar_bar=False)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_key_press_event():
    plotter = pyvista.Plotter()
    plotter.key_press_event(None, None)
    plotter.close()


@skip_no_plotting
def test_enable_picking_gc():
    plotter = pyvista.Plotter()
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere)
    plotter.enable_cell_picking()
    plotter.close()


@skip_no_plotting
def test_left_button_down():
    plotter = pyvista.Plotter()
    if VTK9:
        with pytest.raises(ValueError):
            plotter.left_button_down(None, None)
    else:
        plotter.left_button_down(None, None)
    plotter.close()


@skip_no_plotting
def test_show_axes():
    plotter = pyvista.Plotter()
    plotter.show_axes()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_cell_arrays(sphere):
    plotter = pyvista.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=10, rng=sphere.n_faces, show_scalar_bar=False)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_clim(sphere):
    plotter = pyvista.Plotter()
    scalars = np.arange(sphere.n_faces)
    plotter.add_mesh(sphere, interpolate_before_map=True, scalars=scalars,
                     n_colors=5, clim=10, show_scalar_bar=False)
    plotter.show(before_close_callback=verify_cache_image)
    assert plotter.mapper.GetScalarRange() == (-10, 10)


@skip_no_plotting
def test_invalid_n_arrays(sphere):
    with pytest.raises(ValueError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(sphere, scalars=np.arange(10))
        plotter.show()


@skip_no_plotting
def test_plot_arrow():
    cent = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    pyvista.plot_arrows(cent, direction, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_arrows():
    cent = np.array([[0, 0, 0],
                     [1, 0, 0]])
    direction = np.array([[1, 1, 1],
                          [-1, -1, -1]])
    pyvista.plot_arrows(cent, direction, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_arrows():
    vector = np.array([1, 0, 0])
    center = np.array([0, 0, 0])
    plotter = pyvista.Plotter()
    plotter.add_arrows(cent=center, direction=vector, mag=2.2, color="#009900")
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_axes():
    plotter = pyvista.Plotter()
    plotter.add_orientation_widget(pyvista.Cube())
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_box_axes():
    plotter = pyvista.Plotter()
    plotter.add_axes(box=True, box_args={'color_box': True})
    plotter.add_mesh(pyvista.Sphere())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_screenshot(tmpdir):
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.Sphere())
    img = plotter.screenshot(transparent_background=False)
    assert np.any(img)
    img_again = plotter.screenshot()
    assert np.any(img_again)
    filename = str(tmpdir.mkdir("tmpdir").join('export-graphic.svg'))
    plotter.save_graphic(filename)

    # test window and array size
    w, h = 20, 10
    img = plotter.screenshot(transparent_background=False,
                             window_size=(w, h))
    assert img.shape == (h, w, 3)
    img = plotter.screenshot(transparent_background=True,
                             window_size=(w, h))
    assert img.shape == (h, w, 4)

    # check error before first render
    plotter = pyvista.Plotter(off_screen=False)
    plotter.add_mesh(pyvista.Sphere())
    with pytest.raises(RuntimeError):
        plotter.screenshot()


@skip_no_plotting
@pytest.mark.parametrize('ext', SUPPORTED_FORMATS)
def test_save_screenshot(tmpdir, sphere, ext):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp' + ext))
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.screenshot(filename)
    assert os.path.isfile(filename)
    assert Path(filename).stat().st_size


@skip_no_plotting
def test_scalars_by_name():
    plotter = pyvista.Plotter()
    data = examples.load_uniform()
    plotter.add_mesh(data, scalars='Spatial Cell Data')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_block_plot():
    multi = pyvista.MultiBlock()
    multi.append(examples.load_rectilinear())
    uni = examples.load_uniform()
    arr = np.random.rand(uni.n_cells)
    uni.cell_arrays.append(arr, 'Random Data')
    multi.append(uni)
    # And now add a data set without the desired array and a NULL component
    multi[3] = examples.load_airplane()
    with pytest.raises(ValueError):
        # The scalars are not available in all datasets so raises ValueError
        multi.plot(scalars='Random Data', multi_colors=True)
    multi.plot(multi_colors=True, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_clear(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.clear()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_texture():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = examples.load_globe_texture()
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_texture_alone(tmpdir):
    """"Test adding a texture to a plot"""
    path = str(tmpdir.mkdir("tmpdir"))
    image = Image.new('RGB', (10, 10), color='blue')
    filename = os.path.join(path, 'tmp.jpg')
    image.save(filename)

    texture = pyvista.read_texture(filename)
    texture.plot(rgba=True, before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_texture_associated():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_read_texture_from_numpy():
    """"Test adding a texture to a plot"""
    globe = examples.load_globe()
    texture = pyvista.numpy_to_texture(imageio.imread(examples.mapfile))
    plotter = pyvista.Plotter()
    plotter.add_mesh(globe, texture=texture)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_rgb():
    """"Test adding a texture to a plot"""
    cube = pyvista.Cube()
    cube.clear_arrays()
    x_face_color = (255, 0, 0)
    y_face_color = (0, 255, 0)
    z_face_color = (0, 0, 255)
    face_colors = np.array([x_face_color,
                            x_face_color,
                            y_face_color,
                            y_face_color,
                            z_face_color,
                            z_face_color,
                            ], dtype=np.uint8)
    cube.cell_arrays['face_colors'] = face_colors
    plotter = pyvista.Plotter()
    plotter.add_mesh(cube, scalars='face_colors', rgb=True)
    plotter.show(before_close_callback=verify_cache_image)


def setup_multicomponent_data():
    """Create a dataset with vector values on points and cells."""
    data = pyvista.Plane()

    vector_values_points = np.empty((data.n_points, 3))
    vector_values_points[:, :] = [3., 4., 0.]  # Vector has this value at all points

    vector_values_cells = np.empty((data.n_cells, 3))
    vector_values_cells[:, :] = [3., 4., 0.]  # Vector has this value at all cells

    data['vector_values_points'] = vector_values_points
    data['vector_values_cells'] = vector_values_cells

    return data


@skip_no_plotting
def test_vector_array_with_cells_and_points():
    """Test using vector valued data with and without component arg."""
    data = setup_multicomponent_data()

    # test no component argument
    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_points')
    p.show()

    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_cells')
    p.show()

    # test component argument
    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_points', component=0)
    p.show()

    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_cells', component=0)
    p.show()


@skip_no_plotting
def test_vector_array():
    """Test using vector valued data for image regression."""
    data = setup_multicomponent_data()

    p = pyvista.Plotter(shape=(2, 2))
    p.subplot(0, 0)
    p.add_mesh(data, scalars="vector_values_points", show_scalar_bar=False)
    p.subplot(0, 1)
    p.add_mesh(data.copy(), scalars="vector_values_points", component=0)
    p.subplot(1, 0)
    p.add_mesh(data.copy(), scalars="vector_values_points", component=1)
    p.subplot(1, 1)
    p.add_mesh(data.copy(), scalars="vector_values_points", component=2)
    p.link_views()
    p.show()

    # p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_vector_plotting_doesnt_modify_data():
    """Test that the operations in plotting do not modify the data in the mesh."""
    data = setup_multicomponent_data()

    copy_vector_values_points = data["vector_values_points"].copy()
    copy_vector_values_cells = data["vector_values_cells"].copy()

    # test that adding a vector with no component parameter to a Plotter instance
    # does not modify it.
    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_points')
    p.show()
    assert np.array_equal(data['vector_values_points'], copy_vector_values_points)

    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_cells')
    p.show()
    assert np.array_equal(data['vector_values_cells'], copy_vector_values_cells)

    # test that adding a vector with a component parameter to a Plotter instance
    # does not modify it.
    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_points', component=0)
    p.show()
    assert np.array_equal(data['vector_values_points'], copy_vector_values_points)

    p = pyvista.Plotter()
    p.add_mesh(data, scalars='vector_values_cells', component=0)
    p.show()
    assert np.array_equal(data['vector_values_cells'], copy_vector_values_cells)


@skip_no_plotting
def test_vector_array_fail_with_incorrect_component():
    """Test failure modes of component argument."""
    data = setup_multicomponent_data()

    p = pyvista.Plotter()

    # Non-Integer
    with pytest.raises(TypeError):
        p.add_mesh(data, scalars='vector_values_points', component=1.5)
        p.show()

    # Component doesn't exist
    p = pyvista.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(data, scalars='vector_values_points', component=3)
        p.show()

    # Component doesn't exist
    p = pyvista.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(data, scalars='vector_values_points', component=-1)
        p.show()


@skip_no_plotting
def test_camera(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.view_xy()
    plotter.view_xz()
    plotter.view_yz()
    plotter.add_mesh(examples.load_uniform(), reset_camera=True, culling=True)
    plotter.view_xy(True)
    plotter.view_xz(True)
    plotter.view_yz(True)
    plotter.show(before_close_callback=verify_cache_image)
    plotter.camera_position = None

    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.camera.zoom(5)
    plotter.camera.up = 0, 0, 10
    plotter.show()


@skip_no_plotting
def test_multi_renderers():
    plotter = pyvista.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text('Render Window 0', font_size=30)
    sphere = pyvista.Sphere()
    plotter.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)
    plotter.add_scalar_bar('Z', vertical=True)

    plotter.subplot(0, 1)
    plotter.add_text('Render Window 1', font_size=30)
    plotter.add_mesh(pyvista.Cube(), show_edges=True)

    plotter.subplot(1, 0)
    plotter.add_text('Render Window 2', font_size=30)
    plotter.add_mesh(pyvista.Arrow(), color='y', show_edges=True)

    plotter.subplot(1, 1)
    plotter.add_text('Render Window 3', position=(0., 0.),
                     font_size=30, viewport=True)
    plotter.add_mesh(pyvista.Cone(), color='g', show_edges=True,
                     culling=True)
    plotter.add_bounding_box(render_lines_as_tubes=True, line_width=5)
    plotter.show_bounds(all_edges=True)

    plotter.update_bounds_axes()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_2x1():

    # Test subplot indices (2 rows by 1 column)
    plotter = pyvista.Plotter(shape=(2, 1))
    # First row
    plotter.subplot(0,0)
    plotter.add_mesh(pyvista.Sphere())
    # Second row
    plotter.subplot(1,0)
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_1x2():
    # Test subplot indices (1 row by 2 columns)
    plotter = pyvista.Plotter(shape=(1, 2))
    # First column
    plotter.subplot(0, 0)
    plotter.add_mesh(pyvista.Sphere())
    # Second column
    plotter.subplot(0, 1)
    plotter.add_mesh(pyvista.Cube())
    plotter.show(before_close_callback=verify_cache_image)

@skip_no_plotting
def test_multi_renderers_bad_indices():
    with pytest.raises(IndexError):
        # Test bad indices
        plotter = pyvista.Plotter(shape=(1, 2))
        plotter.subplot(0, 0)
        plotter.add_mesh(pyvista.Sphere())
        plotter.subplot(1, 0)
        plotter.add_mesh(pyvista.Cube())
        plotter.show()


@skip_no_plotting
def test_multi_renderers_subplot_ind_3x1():
    # Test subplot 3 on left, 1 on right
    plotter = pyvista.Plotter(shape='3|1')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_3x1_splitting_pos():
    # Test subplot 3 on top, 1 on bottom
    plotter = pyvista.Plotter(shape='3/1', splitting_position=0.5)
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_multi_renderers_subplot_ind_1x3():
    # Test subplot 3 on bottom, 1 on top
    plotter = pyvista.Plotter(shape='1|3')
    # First column
    plotter.subplot(0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(2)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(3)
    plotter.add_mesh(pyvista.Cone())
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_subplot_groups():
    plotter = pyvista.Plotter(shape=(3, 3), groups=[(1, [1, 2]), (np.s_[:], 0)])
    plotter.subplot(0, 0)
    plotter.add_mesh(pyvista.Sphere())
    plotter.subplot(0, 1)
    plotter.add_mesh(pyvista.Cube())
    plotter.subplot(0, 2)
    plotter.add_mesh(pyvista.Arrow())
    plotter.subplot(1, 1)
    plotter.add_mesh(pyvista.Cylinder())
    plotter.subplot(2, 1)
    plotter.add_mesh(pyvista.Cone())
    plotter.subplot(2, 2)
    plotter.add_mesh(pyvista.Box())
    plotter.show(before_close_callback=verify_cache_image)


def test_subplot_groups_fail():
    # Test group overlap
    with pytest.raises(ValueError):
        # Partial overlap
        pyvista.Plotter(shape=(3, 3), groups=[([1, 2], [0, 1]), ([0, 1], [1, 2])])
    with pytest.raises(ValueError):
        # Full overlap (inner)
        pyvista.Plotter(shape=(4, 4), groups=[(np.s_[:], np.s_[:]), ([1, 2], [1, 2])])
    with pytest.raises(ValueError):
        # Full overlap (outer)
        pyvista.Plotter(shape=(4, 4), groups=[(1, [1, 2]), ([0, 3], np.s_[:])])


@skip_no_plotting
@skip_windows_dev_whl
def test_link_views(sphere):
    plotter = pyvista.Plotter(shape=(1, 4))
    plotter.subplot(0, 0)
    plotter.add_mesh(sphere, smooth_shading=False, show_edges=False)
    plotter.subplot(0, 1)
    plotter.add_mesh(sphere, smooth_shading=True, show_edges=False)
    plotter.subplot(0, 2)
    plotter.add_mesh(sphere, smooth_shading=False, show_edges=True)
    plotter.subplot(0, 3)
    plotter.add_mesh(sphere, smooth_shading=True, show_edges=True)
    with pytest.raises(TypeError):
        plotter.link_views(views='foo')
    plotter.link_views([0, 1])
    plotter.link_views()
    with pytest.raises(TypeError):
        plotter.unlink_views(views='foo')
    plotter.unlink_views([0, 1])
    plotter.unlink_views(2)
    plotter.unlink_views()
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_orthographic_slicer(uniform):
    uniform.set_active_scalars('Spatial Cell Data')
    slices = uniform.slice_orthogonal()

    # Orthographic Slicer
    p = pyvista.Plotter(shape=(2, 2))

    p.subplot(1, 1)
    p.add_mesh(slices, clim=uniform.get_data_range())
    p.add_axes()
    p.enable()

    p.subplot(0, 0)
    p.add_mesh(slices['XY'])
    p.view_xy()
    p.disable()

    p.subplot(0, 1)
    p.add_mesh(slices['XZ'])
    p.view_xz(negative=True)
    p.disable()

    p.subplot(1, 0)
    p.add_mesh(slices['YZ'])
    p.view_yz()
    p.disable()

    p.show()


@skip_no_plotting
def test_remove_actor(uniform):
    plotter = pyvista.Plotter()
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.add_mesh(uniform.copy(), name='data')
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_image_properties():
    mesh = examples.load_uniform()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.show(auto_close=False)  # DO NOT close plotter
    # Get RGB image
    _ = p.image
    # Get the depth image
    _ = p.get_image_depth()
    p.close()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.store_image = True
    assert p.store_image is True
    p.show()  # close plotter
    # Get RGB image
    _ = p.image
    # verify property matches method while testing both available
    assert np.allclose(p.image_depth, p.get_image_depth(), equal_nan=True)
    p.close()

    # gh-920
    rr = np.array(
        [[-0.5, -0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, -0.5, 1]])
    tris = np.array([[3, 0, 2, 1], [3, 2, 0, 3]])
    mesh = pyvista.PolyData(rr, tris)
    p = pyvista.Plotter()
    p.add_mesh(mesh, color=True)
    p.renderer.camera_position = (0., 0., 1.)
    p.renderer.ResetCamera()
    p.enable_parallel_projection()
    assert p.renderer.camera_set
    p.show(interactive=False, auto_close=False)
    img = p.get_image_depth(fill_value=0.)
    rng = np.ptp(img)
    assert 0.3 < rng < 0.4, rng  # 0.3313504 in testing
    p.close()


@skip_no_plotting
def test_volume_rendering():
    # Really just making sure no errors are thrown
    vol = examples.load_uniform()
    vol.plot(volume=True, opacity='linear')

    plotter = pyvista.Plotter()
    plotter.add_volume(vol, opacity='sigmoid', cmap='jet', n_colors=15)
    plotter.show()

    # Now test MultiBlock rendering
    data = pyvista.MultiBlock(dict(a=examples.load_uniform(),
                                   b=examples.load_uniform(),
                                   c=examples.load_uniform(),
                                   d=examples.load_uniform(),))
    data['a'].rename_array('Spatial Point Data', 'a')
    data['b'].rename_array('Spatial Point Data', 'b')
    data['c'].rename_array('Spatial Point Data', 'c')
    data['d'].rename_array('Spatial Point Data', 'd')
    data.plot(volume=True, multi_colors=True)

    # Check that NumPy arrays work
    arr = vol["Spatial Point Data"].reshape(vol.dimensions)
    pyvista.plot(arr, volume=True, opacity='linear')


@skip_no_plotting
def test_plot_compare_four():
    # Really just making sure no errors are thrown
    mesh = examples.load_uniform()
    data_a = mesh.contour()
    data_b = mesh.threshold_percent(0.5)
    data_c = mesh.decimate_boundary(0.5)
    data_d = mesh.glyph(scale=False)
    pyvista.plot_compare_four(data_a, data_b, data_c, data_d,
                              disply_kwargs={'color': 'w'},
                              show_kwargs={'before_close_callback': verify_cache_image})


@skip_no_plotting
def test_plot_depth_peeling():
    mesh = examples.load_airplane()
    p = pyvista.Plotter()
    p.add_mesh(mesh)
    p.enable_depth_peeling()
    p.disable_depth_peeling()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plot(airplane):
    airplane.plot(eye_dome_lighting=True, before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_plotter(airplane):
    p = pyvista.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason="No testing on windows for EDL")
def test_plot_eye_dome_lighting_enable_disable(airplane):
    p = pyvista.Plotter()
    p.add_mesh(airplane)
    p.enable_eye_dome_lighting()
    p.disable_eye_dome_lighting()
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array(uniform):
    # Test with opacity array
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['opac'] = opac
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='opac')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array_uncertainty(uniform):
    # Test with uncertainty array (transparency)
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity='unc',
               use_transparency=True)
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_opacity_by_array_user_transform(uniform):
    uniform['Spatial Point Data'] /= uniform['Spatial Point Data'].max()

    # Test with user defined transfer function
    opacities = [0, 0.2, 0.9, 0.2, 0.1]
    p = pyvista.Plotter()
    p.add_mesh(uniform, scalars='Spatial Point Data', opacity=opacities)
    p.show()  # note: =verify_cache_image does not work between Xvfb


def test_opacity_mismatched_fail(uniform):
    opac = uniform['Spatial Point Data'] / uniform['Spatial Point Data'].max()
    uniform['unc'] = opac

    # Test using mismatched arrays
    p = pyvista.Plotter()
    with pytest.raises(ValueError):
        p.add_mesh(uniform, scalars='Spatial Cell Data', opacity='unc')


def test_opacity_transfer_functions():
    n = 256
    mapping = pyvista.opacity_transfer_function('linear', n)
    assert len(mapping) == n
    mapping = pyvista.opacity_transfer_function('sigmoid_10', n)
    assert len(mapping) == n
    with pytest.raises(KeyError):
        mapping = pyvista.opacity_transfer_function('foo', n)
    with pytest.raises(RuntimeError):
        mapping = pyvista.opacity_transfer_function(np.linspace(0, 1, 2*n), n)
    foo = np.linspace(0, n, n)
    mapping = pyvista.opacity_transfer_function(foo, n)
    assert np.allclose(foo, mapping)
    foo = [0, 0.2, 0.9, 0.2, 0.1]
    mapping = pyvista.opacity_transfer_function(foo, n, interpolate=False)
    assert len(mapping) == n
    foo = [3, 5, 6, 10]
    mapping = pyvista.opacity_transfer_function(foo, n)
    assert len(mapping) == n


@skip_no_plotting
def test_closing_and_mem_cleanup():
    n = 5
    for _ in range(n):
        for _ in range(n):
            p = pyvista.Plotter()
            for k in range(n):
                p.add_mesh(pyvista.Sphere(radius=k))
            p.show()
        pyvista.close_all()


@skip_no_plotting
def test_above_below_scalar_range_annotations():
    p = pyvista.Plotter()
    p.add_mesh(examples.load_uniform(), clim=[100, 500], cmap='viridis',
               below_color='blue', above_color='red')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_user_annotations_scalar_bar_mesh(uniform):
    p = pyvista.Plotter()
    p.add_mesh(uniform, annotations={100.: 'yum'})
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_user_annotations_scalar_bar_volume(uniform):
    p = pyvista.Plotter()
    p.add_volume(uniform, annotations={100.: 'yum'})
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_scalar_bar_args_unmodified_add_mesh(sphere):
    sargs = {"vertical": True}
    sargs_copy = sargs.copy()

    p = pyvista.Plotter()
    p.add_mesh(sphere, scalar_bar_args=sargs)

    assert sargs == sargs_copy


@skip_no_plotting
def test_scalar_bar_args_unmodified_add_volume(uniform):
    sargs = {"vertical": True}
    sargs_copy = sargs.copy()

    p = pyvista.Plotter()
    p.add_volume(uniform, scalar_bar_args=sargs)

    assert sargs == sargs_copy


@skip_no_plotting
def test_plot_string_array():
    mesh = examples.load_uniform()
    labels = np.empty(mesh.n_cells, dtype='<U10')
    labels[:] = 'High'
    labels[mesh['Spatial Cell Data'] < 300] = 'Medium'
    labels[mesh['Spatial Cell Data'] < 100] = 'Low'
    mesh['labels'] = labels
    p = pyvista.Plotter()
    p.add_mesh(mesh, scalars='labels')
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_fail_plot_table():
    """Make sure tables cannot be plotted"""
    table = pyvista.Table(np.random.rand(50, 3))
    with pytest.raises(TypeError):
        pyvista.plot(table)
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(table)


@skip_no_plotting
def test_bad_keyword_arguments():
    """Make sure bad keyword arguments raise an error"""
    mesh = examples.load_uniform()
    with pytest.raises(TypeError):
        pyvista.plot(mesh, foo=5)
    with pytest.raises(TypeError):
        pyvista.plot(mesh, scalar=mesh.active_scalars_name)
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(mesh, scalar=mesh.active_scalars_name)
        plotter.show()
    with pytest.raises(TypeError):
        plotter = pyvista.Plotter()
        plotter.add_mesh(mesh, foo="bad")
        plotter.show()


@skip_no_plotting
def test_cmap_list(sphere):
    n = sphere.n_points
    scalars = np.empty(n)
    scalars[:n//3] = 0
    scalars[n//3:2*n//3] = 1
    scalars[2*n//3:] = 2

    with pytest.raises(TypeError):
        sphere.plot(scalars=scalars, cmap=['red', None, 'blue'])

    sphere.plot(scalars=scalars, cmap=['red', 'green', 'blue'],
                before_close_callback=verify_cache_image)


@skip_no_plotting
def test_default_name_tracking():
    N = 10
    color = "tan"

    p = pyvista.Plotter()
    for i in range(N):
        for j in range(N):
            center = (i, j, 0)
            mesh = pyvista.Sphere(center=center)
            p.add_mesh(mesh, color=color)
    n_made_it = len(p.renderer._actors)
    p.show()
    assert n_made_it == N**2

    # release attached scalars
    mesh.ReleaseData()
    del mesh


@skip_no_plotting
def test_add_background_image_global(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=True)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_background_image_not_global(sphere):
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_background_image(examples.mapfile, as_global=False)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_background_image_subplots(airplane):
    pl = pyvista.Plotter(shape=(2, 2))
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(airplane)
    pl.subplot(1, 1)
    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.add_mesh(airplane)
    pl.remove_background_image()

    # should error out as there's no background
    with pytest.raises(RuntimeError):
        pl.remove_background_image()

    pl.add_background_image(examples.mapfile, scale=1, as_global=False)
    pl.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_add_remove_floor(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.add_bounding_box()  # needed for update_bounds_axes
    assert len(pl.renderer._floors) == 1
    pl.add_mesh(pyvista.Sphere(1.0))
    pl.update_bounds_axes()
    assert len(pl.renderer._floors) == 1
    pl.show()

    pl = pyvista.Plotter()
    pl.add_mesh(sphere)
    pl.add_floor(color='b', line_width=2, lighting=True)
    pl.remove_floors()
    assert not pl.renderer._floors
    pl.show(before_close_callback=verify_cache_image)


def test_reset_camera_clipping_range(sphere):
    pl = pyvista.Plotter()
    pl.add_mesh(sphere)

    # get default clipping range
    default_clipping_range = pl.camera.clipping_range

    # make sure we assign something different than default
    assert default_clipping_range != (10, 100)

    # set clipping range to some random numbers and make sure
    # assignment is successful
    pl.camera.clipping_range = (10, 100)
    assert pl.camera.clipping_range == (10, 100)

    pl.reset_camera_clipping_range()
    assert pl.camera.clipping_range == default_clipping_range
    assert pl.camera.clipping_range != (10, 100)


def test_index_vs_loc():
    # first: 2d grid
    pl = pyvista.Plotter(shape=(2, 3))
    # index_to_loc valid cases
    vals = [0, 2, 4]
    expecteds = [(0, 0), (0, 2), (1, 1)]
    for val, expected in zip(vals, expecteds):
        assert tuple(pl.renderers.index_to_loc(val)) == expected
    # loc_to_index valid cases
    vals = [(0, 0), (0, 2), (1, 1)]
    expecteds = [0, 2, 4]
    for val, expected in zip(vals, expecteds):
        assert pl.renderers.loc_to_index(val) == expected
        assert pl.renderers.loc_to_index(expected) == expected

    # indexing failing cases
    with pytest.raises(TypeError):
        pl.renderers.index_to_loc(1.5)
    with pytest.raises(IndexError):
        pl.renderers.index_to_loc((-1))
    with pytest.raises(TypeError):
        pl.renderers.index_to_loc((1, 2))
    with pytest.raises(IndexError):
        pl.renderers.loc_to_index((-1, 0))
    with pytest.raises(IndexError):
        pl.renderers.loc_to_index((0, -1))
    with pytest.raises(TypeError):
        pl.renderers.loc_to_index({1, 2})
    with pytest.raises(ValueError):
        pl.renderers.loc_to_index((1, 2, 3))

    # set active_renderer fails
    with pytest.raises(IndexError):
        pl.renderers.set_active_renderer(0, -1)

    # then: "1d" grid
    pl = pyvista.Plotter(shape='2|3')
    # valid cases
    for val in range(5):
        assert pl.renderers.index_to_loc(val) == val
        assert pl.renderers.index_to_loc(np.int_(val)) == val
        assert pl.renderers.loc_to_index(val) == val
        assert pl.renderers.loc_to_index(np.int_(val)) == val


@skip_no_plotting
def test_interactive_update():
    # Regression test for #1053
    p = pyvista.Plotter()
    p.show(interactive_update=True)
    assert isinstance(p.iren.interactor, vtk.vtkRenderWindowInteractor)
    p.close()

    p = pyvista.Plotter()
    with pytest.warns(UserWarning):
        p.show(auto_close=True, interactive_update=True)


def test_where_is():
    plotter = pyvista.Plotter(shape=(2, 2))
    plotter.subplot(0, 0)
    plotter.add_mesh(pyvista.Box(), name='box')
    plotter.subplot(0, 1)
    plotter.add_mesh(pyvista.Sphere(), name='sphere')
    plotter.subplot(1, 0)
    plotter.add_mesh(pyvista.Box(), name='box')
    plotter.subplot(1, 1)
    plotter.add_mesh(pyvista.Cone(), name='cone')
    places = plotter.where_is('box')
    assert isinstance(places, list)
    for loc in places:
        assert isinstance(loc, tuple)


@skip_no_plotting
def test_log_scale():
    mesh = examples.load_uniform()
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh, log_scale=True)
    plotter.show()


@skip_no_plotting
def test_set_focus():
    plane = pyvista.Plane()
    p = pyvista.Plotter()
    p.add_mesh(plane, color="tan", show_edges=True)
    p.set_focus((1, 0, 0))
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_set_viewup():
    plane = pyvista.Plane()
    plane_higher = pyvista.Plane(center=(0, 0, 1), i_size=0.5, j_size=0.5)
    p = pyvista.Plotter()
    p.add_mesh(plane, color="tan", show_edges=False)
    p.add_mesh(plane_higher, color="red", show_edges=False)
    p.set_viewup((1.0, 1.0, 1.0))
    p.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_remove_scalar_bar(sphere):
    sphere['z'] = sphere.points[:, 2]
    plotter = pyvista.Plotter()
    plotter.add_mesh(sphere, show_scalar_bar=False)
    plotter.add_scalar_bar(interactive=True)
    assert len(plotter.scalar_bars) == 1
    plotter.remove_scalar_bar()
    assert len(plotter.scalar_bars) == 0
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_shadows():
    plotter = pyvista.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pyvista.Plane(center=(0, plane_y, 0),
                               direction=(0, 1, 0),
                               i_size=5, j_size=5)
        plotter.add_mesh(screen, color='white')

    light = pyvista.Light(position=(0, 0, 0), focal_point=(0, 1, 0),
                          color='cyan', intensity=15, cone_angle=15,
                          positional=True, show_actor=True,
                          attenuation_values=(2, 0, 0))

    plotter.add_light(light)
    plotter.view_vector((1, -2, 2))

    # verify disabling shadows when not enabled does nothing
    plotter.disable_shadows()

    plotter.enable_shadows()

    # verify shadows can safely be enabled twice
    plotter.enable_shadows()

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_shadows_enable_disable():
    """Test shadows are added and removed properly"""
    plotter = pyvista.Plotter(lighting=None)

    # add several planes
    for plane_y in [2, 5, 10]:
        screen = pyvista.Plane(center=(0, plane_y, 0),
                               direction=(0, 1, 0),
                               i_size=5, j_size=5)
        plotter.add_mesh(screen, color='white')

    light = pyvista.Light(position=(0, 0, 0), focal_point=(0, 1, 0),
                          color='cyan', intensity=15, cone_angle=15)
    light.positional = True
    light.attenuation_values = (2, 0, 0)
    light.show_actor()

    plotter.add_light(light)
    plotter.view_vector((1, -2, 2))

    # add and remove and verify that the light passes through all via
    # image cache
    plotter.enable_shadows()
    plotter.disable_shadows()

    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_lighting_change_positional_true_false(sphere):
    light = pyvista.Light(positional=True, show_actor=True)

    plotter = pyvista.Plotter(lighting=None)
    plotter.add_light(light)
    light.positional = False
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plot_lighting_change_positional_false_true(sphere):
    light = pyvista.Light(positional=False, show_actor=True)

    plotter = pyvista.Plotter(lighting=None)

    plotter.add_light(light)
    light.positional = True
    plotter.add_mesh(sphere)
    plotter.show(before_close_callback=verify_cache_image)


@skip_no_plotting
def test_plotter_image():
    plotter = pyvista.Plotter()
    plotter.show()
    with pytest.raises(AttributeError, match='To retrieve an image after'):
        plotter.image

    plotter = pyvista.Plotter()
    wsz = tuple(plotter.window_size)
    plotter.store_image = True
    plotter.show()
    assert plotter.image.shape[:2] == wsz
